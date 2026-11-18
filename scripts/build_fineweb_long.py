"""Build a reproducible long-document corpus from a pinned default length index.

Stages: plan -> extract (parallel ranks) -> select -> materialize -> verify -> upload.
The shared artifact contains text. Local uint32 token files preserve document boundaries.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import unicodedata
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tokenizers as tokenizers_package
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from profile_fineweb_edu_lengths import sha256, write_json
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("DYNAMIC_SEQ_LEN_ASSETS", ROOT / "assets"))
TOKENIZER = ASSETS / "hf/llama3-tokenizer"
THRESHOLDS = (32768, 28672, 24576, 20480, 16384, 12288, 8192)


def tokenizer_provenance():
    """Read the same provenance from local preparation or the shared package."""
    path = TOKENIZER / "tokenizer_source.json"
    if not path.is_file():
        path = ASSETS / "data/fineweb_edu_long_30b_shuffled_v1/tokenizer_provenance.json"
    info = json.loads(path.read_text())
    for name, expected in info["files_sha256"].items():
        if sha256(TOKENIZER / name) != expected:
            raise ValueError(f"Tokenizer provenance checksum mismatch: {name}")
    return info


def identity(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def normalized_hash(text):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFC", text).split()).encode()).hexdigest()


def url_group(url, doc_id):
    try:
        parts = urlsplit(url or "")
        host = (parts.hostname or "").lower().removeprefix("www.")
    except ValueError:
        return hashlib.sha256((url or doc_id).encode()).hexdigest()
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    canonical = host + parts.path.rstrip("/") + ("?" + urlencode(sorted(query)) if query else "")
    return hashlib.sha256((canonical if host else doc_id).encode()).hexdigest()


def load_plan(root):
    plan = json.loads((root / "extraction_plan.json").read_text())
    expected = plan.pop("plan_sha256")
    if identity(plan) != expected:
        raise ValueError("Extraction plan identity mismatch")
    plan["plan_sha256"] = expected
    return plan


def plan_stage(args):
    source = json.loads((args.root / "profile/indexed_manifest.json").read_text())
    tok_info = tokenizer_provenance()
    digest = sha256(TOKENIZER / "tokenizer.json")
    assert digest == tok_info["files_sha256"]["tokenizer.json"]
    # Sample source shards in a reproducible random order, rather than crawl order.
    files = sorted(source["files"], key=lambda e: hashlib.sha256(("long-v1:" + e["path"]).encode()).digest())
    if args.ranks > len(files):
        raise ValueError("There must be at least one source file per rank")
    plan = {"schema_version": 1, "repo_id": source["repo_id"], "revision": source["revision"],
            "config": "default", "seed": "long-v1", "ranks": args.ranks,
            "source_minimum": args.minimum, "llama_minimum": args.minimum,
            "source_maximum_exclusive": args.source_maximum,
            "candidate_token_budget_per_rank": args.candidate_tokens // args.ranks,
            "train_token_target": args.train_tokens, "validation_minimum_tokens": 10_000_000,
            "validation_hash_modulus": 10000,
            "validation_hash_cutoff": 20, "tokenizer": tok_info,
            "tokenizer_sha256": digest, "encoding": "raw text; add_special_tokens=False; no truncation or padding",
            "implementation": {"builder_sha256": sha256(Path(__file__)), "python": sys.version,
                               "tokenizers": tokenizers_package.__version__, "pyarrow": pa.__version__,
                               "numpy": np.__version__},
            "source_candidate_note": f"Source token_count >= {args.minimum} is a retrieval prefilter. "
                                     + (f"Source token_count must also be below {args.source_maximum}. " if args.source_maximum else "")
                                     + "This is not an exhaustive Llama-length classification of default.",
            "files": [{**entry, "extract_rank": i % args.ranks, "order": i} for i, entry in enumerate(files)]}
    plan["plan_sha256"] = identity(plan)
    path = args.root / "extraction_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Refusing to replace a different extraction plan")
    write_json(path, plan)
    print(json.dumps({"event": "plan", "ranks": args.ranks, "candidate_tokens": args.candidate_tokens,
                      "train_target": args.train_tokens, "plan_sha256": plan["plan_sha256"]}))


def candidate_schema():
    return pa.schema([
        ("id", pa.string()), ("text", pa.large_string()), ("url", pa.string()),
        ("source_tokens", pa.int64()), ("llama3_tokens", pa.int64()),
        ("text_sha256", pa.string()), ("normalized_sha256", pa.string()), ("url_group", pa.string()),
        ("source_path", pa.string()), ("source_row", pa.int64()), ("dump", pa.string()),
        ("edu_score", pa.float32()), ("input_ids", pa.large_list(pa.uint32()))])


@contextmanager
def staged_source(entry, plan):
    """Stage one shard on the compute node; avoid thousands of small WAN reads."""
    meta = get_hf_file_metadata(hf_hub_url(plan["repo_id"], entry["path"], repo_type="dataset",
                                        revision=plan["revision"]), token=False, timeout=30)
    if meta.commit_hash != plan["revision"] or meta.size != entry["size"]:
        raise ValueError("Remote source identity mismatch")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fineweb-long-", dir=os.environ.get("TMPDIR")) as folder:
        path = Path(folder) / "source.parquet"
        size = entry["size"]
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(descriptor, size)
        def transfer(bounds):
            first, last = bounds
            for attempt in range(3):
                try:
                    request = Request(meta.location, headers={"Range": f"bytes={first}-{last}"})
                    with urlopen(request, timeout=120) as remote:
                        if remote.status != 206 or remote.headers.get("Content-Range") != f"bytes {first}-{last}/{size}":
                            raise ValueError("Server returned an unexpected byte range")
                        offset = first
                        while block := remote.read(4 * 1024**2):
                            if offset + len(block) > last + 1:
                                raise ValueError("Source range is larger than requested")
                            view = memoryview(block)
                            while view:
                                written = os.pwrite(descriptor, view, offset)
                                view = view[written:]
                                offset += written
                        if offset != last + 1:
                            raise ValueError("Incomplete source range")
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        try:
            chunk_size = 64 * 1024**2
            ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(transfer, ranges))
        finally:
            os.close(descriptor)
        if path.stat().st_size != size or (entry.get("sha256") and sha256(path) != entry["sha256"]):
            raise ValueError("Downloaded source checksum mismatch")
        print(json.dumps({"event": "source_staged", "source": entry["path"], "bytes": size,
                          "seconds": time.perf_counter() - started}), flush=True)
        yield path


def extract_file(args, plan, entry, tokenizer):
    key = f"{entry['order']:05d}"
    folder = args.root / "candidates"
    folder.mkdir(exist_ok=True)
    final = folder / f"{key}.parquet"
    record = folder / f"{key}.json"
    if record.exists():
        result = json.loads(record.read_text())
        if result["plan_sha256"] != plan["plan_sha256"] or sha256(final) != result["sha256"]:
            raise ValueError("Candidate cache checksum or plan mismatch")
        return result
    profile = json.loads((args.root / entry["profile_record"]).read_text())
    index_path = args.root / entry["length_index"]
    if sha256(index_path) != profile["length_index_sha256"]:
        raise ValueError("Source length index checksum mismatch")
    lengths = np.load(index_path, mmap_mode="r")
    temporary = final.with_suffix(".partial.parquet")
    started = time.perf_counter()
    for attempt in range(3):
        count = token_total = selected_source = 0
        try:
            with staged_source(entry, plan) as local_source, pq.ParquetWriter(
                temporary, candidate_schema(), compression="zstd", use_dictionary=False,
                write_statistics=True) as writer:
                source = pq.ParquetFile(local_source, pre_buffer=False)
                if len(lengths) != source.metadata.num_rows:
                    raise ValueError("Source row count changed")
                columns = [c for c in ["id", "text", "url", "token_count", "dump", "score"]
                           if c in source.schema_arrow.names]
                offset = 0
                for group in range(source.metadata.num_row_groups):
                    size = source.metadata.row_group(group).num_rows
                    eligible = lengths[offset:offset + size] >= plan["source_minimum"]
                    if plan.get("source_maximum_exclusive") is not None:
                        eligible &= lengths[offset:offset + size] < plan["source_maximum_exclusive"]
                    indices = np.flatnonzero(eligible)
                    if len(indices):
                        table = source.read_row_group(group, columns=columns, use_threads=False).take(pa.array(indices))
                        docs = table.to_pylist()
                        for start in range(0, len(docs), 32):
                            batch = docs[start:start + 32]
                            encoded = tokenizer.encode_batch([d["text"] for d in batch], add_special_tokens=False)
                            output = []
                            for j, (doc, enc) in enumerate(zip(batch, encoded)):
                                row = offset + int(indices[start + j])
                                if doc["token_count"] != int(lengths[row]):
                                    raise ValueError("Source token count changed")
                                ids = enc.ids
                                if len(ids) < plan["llama_minimum"]:
                                    continue
                                text = doc["text"]
                                doc_id = str(doc["id"])
                                output.append({"id": doc_id, "text": text, "url": doc["url"],
                                               "source_tokens": doc["token_count"], "llama3_tokens": len(ids),
                                               "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                               "normalized_sha256": normalized_hash(text),
                                               "url_group": url_group(doc["url"], doc_id),
                                               "source_path": entry["path"], "source_row": row,
                                               "dump": doc.get("dump", entry["path"].split("/")[1]),
                                               "edu_score": doc.get("score"), "input_ids": ids})
                                count += 1
                                token_total += len(ids)
                                selected_source += doc["token_count"]
                            if output:
                                writer.write_table(pa.Table.from_pylist(output, schema=candidate_schema()), row_group_size=32)
                    offset += size
                    if group and group % 100 == 0:
                        print(json.dumps({"event": "extract_progress", "rank": args.rank,
                                          "source": entry["path"], "row_groups": group + 1,
                                          "documents": count, "llama3_tokens": token_total}), flush=True)
            temporary.replace(final)
            result = {"file": str(final.relative_to(args.root)), "source_path": entry["path"],
                      "order": entry["order"], "rank": entry["extract_rank"], "documents": count,
                      "llama3_tokens": token_total, "source_tokens": selected_source,
                      "sha256": sha256(final), "plan_sha256": plan["plan_sha256"],
                      "seconds": time.perf_counter() - started}
            write_json(record, result)
            return result
        except Exception as exc:
            print(json.dumps({"event": "retry", "source": entry["path"], "attempt": attempt + 1,
                              "error_type": type(exc).__name__}), flush=True)
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def extract_stage(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run extraction in a CPU allocation")
    plan = load_plan(args.root)
    if not 0 <= args.rank < plan["ranks"]:
        raise ValueError("Invalid extraction rank")
    if sha256(TOKENIZER / "tokenizer.json") != plan["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tokenizer = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    files = [e for e in plan["files"] if e["extract_rank"] == args.rank]
    total = documents = 0
    used = []
    started = time.perf_counter()
    folder = args.root / "extract_status"
    folder.mkdir(exist_ok=True)
    for entry in files:
        record = extract_file(args, plan, entry, tokenizer)
        total += record["llama3_tokens"]
        documents += record["documents"]
        used.append(record["file"])
        done = total >= plan["candidate_token_budget_per_rank"] and not args.exhaust_source
        progress = {"rank": args.rank, "job_id": os.environ["SLURM_JOB_ID"], "files": used,
                    "documents": documents, "llama3_tokens": total, "complete": done,
                    "plan_sha256": plan["plan_sha256"], "seconds": time.perf_counter() - started,
                    "exhaust_source_requested": args.exhaust_source}
        write_json(folder / f"rank-{args.rank:02d}.json", progress)
        print(json.dumps({k: v for k, v in progress.items() if k != "files"}), flush=True)
        if done:
            return
    # A random shard partition can contain fewer long tokens than another rank.
    # Exhausting its complete source assignment is valid; selection enforces the
    # actual global train/validation budgets after deduplication.
    progress.update(complete=True, exhausted_sources=True)
    write_json(folder / f"rank-{args.rank:02d}.json", progress)
    print(json.dumps({k: v for k, v in progress.items() if k != "files"}), flush=True)


def candidate_files(root, plan):
    if plan.get("candidate_snapshot"):
        files = []
        for item in plan["candidate_snapshot"]:
            record = json.loads((root / item["record"]).read_text())
            if (record["sha256"], record["plan_sha256"]) != (item["sha256"], item["extraction_plan_sha256"]):
                raise ValueError("A frozen candidate record changed")
            if not (root / item["file"]).is_file():
                raise ValueError("A frozen candidate file is missing")
            files.append(item["file"])
        if len(files) != len(set(files)):
            raise ValueError("Repeated frozen candidate file")
        return sorted(files)
    if plan.get("candidate_sources"):
        files = []
        for component in plan["candidate_sources"]:
            source = Path(component["root"])
            source_plan = load_plan(source)
            if source_plan["plan_sha256"] != component["plan_sha256"]:
                raise ValueError("A component extraction plan changed")
            files.extend(os.path.relpath(source / name, root) for name in candidate_files(source, source_plan))
        if len(files) != len(set(files)):
            raise ValueError("Repeated candidate source")
        return sorted(files)
    files = []
    for rank in range(plan["ranks"]):
        state = json.loads((root / "extract_status" / f"rank-{rank:02d}.json").read_text())
        if not state["complete"] or state["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Incomplete extraction rank")
        files.extend(state["files"])
    if len(files) != len(set(files)):
        raise ValueError("Repeated candidate file")
    return sorted(files)


def combine_stage(args):
    """Reuse completed candidate pools without copying or encoding their texts."""
    roots = [root.resolve() for root in args.candidate_root or []]
    if len(roots) < 2 or len(roots) != len(set(roots)) or args.root in roots:
        raise ValueError("Supply at least two distinct candidate roots and a separate output root")
    plans = [load_plan(root) for root in roots]
    snapshot = []
    for root, plan in zip(roots, plans):
        if getattr(args, "snapshot", False):
            # A record appears only after its Parquet file is closed and hashed.
            # Unfinished files have no record and cannot enter this snapshot.
            for path in sorted((root / "candidates").glob("*.json")):
                record = json.loads(path.read_text())
                if record["plan_sha256"] != plan["plan_sha256"]:
                    raise ValueError("Candidate record belongs to another extraction plan")
                candidate = root / record["file"]
                if not candidate.is_file():
                    raise ValueError("Completed candidate file is missing")
                snapshot.append({"file": os.path.relpath(candidate, args.root),
                                 "record": os.path.relpath(path, args.root), "sha256": record["sha256"],
                                 "extraction_plan_sha256": record["plan_sha256"],
                                 "source_path": record.get("source_path")})
        else:
            candidate_files(root, plan)
        for key in ["repo_id", "revision", "tokenizer_sha256", "encoding",
                    "validation_hash_modulus", "validation_hash_cutoff"]:
            if plan[key] != plans[0][key]:
                raise ValueError(f"Candidate pools differ in {key}")
    combined = {key: value for key, value in plans[0].items() if key != "plan_sha256"}
    combined.update(candidate_sources=[{"root": str(root), "plan_sha256": plan["plan_sha256"],
                                        "source_minimum": plan.get("source_minimum"),
                                        "source_maximum_exclusive": plan.get("source_maximum_exclusive"),
                                        "llama_minimum": plan.get("llama_minimum")}
                                       for root, plan in zip(roots, plans)],
                    source_minimum=min(plan.get("source_minimum", 0) for plan in plans),
                    source_maximum_exclusive=None,
                    llama_minimum=min(plan.get("llama_minimum", 0) for plan in plans),
                    source_candidate_note="This dataset combines candidate pools with different source-length filters. "
                                          "The manifest records each filter. Llama counts describe these pools, not all default documents.")
    if getattr(args, "snapshot", False):
        if not snapshot:
            raise ValueError("No completed candidate files are available")
        combined["candidate_snapshot"] = sorted(snapshot, key=lambda item: item["file"])
        combined["source_candidate_note"] += " Selection uses a fixed snapshot of completed files. Unfinished files do not enter this snapshot."
    combined["plan_sha256"] = identity(combined)
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root / "extraction_plan.json"
    if path.exists() and json.loads(path.read_text()) != combined:
        raise ValueError("Refusing to replace a different combined plan")
    write_json(path, combined)
    print(json.dumps({"event": "combined", "candidate_roots": list(map(str, roots)),
                      "plan_sha256": combined["plan_sha256"]}), flush=True)


def select_stage(args):
    plan = load_plan(args.root)
    files = candidate_files(args.root, plan)
    db_path = args.root / "selection.sqlite"
    selection_path = args.root / "selection.json"
    if selection_path.exists():
        cached = json.loads(selection_path.read_text())
        if cached["plan_sha256"] != plan["plan_sha256"] or sha256(db_path) != cached["selection_sqlite_sha256"]:
            raise ValueError("Existing selection is inconsistent; inspect before resuming")
        print(json.dumps({"event": "selection_reused", "minimum_llama3_tokens": cached["minimum_llama3_tokens"]}), flush=True)
        return
    local_database = tempfile.TemporaryDirectory(prefix="fineweb-selection-", dir=os.environ.get("TMPDIR"))
    temporary = Path(local_database.name) / "selection.sqlite"
    db = sqlite3.connect(temporary)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-1048576")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("CREATE TABLE docs (doc_id TEXT UNIQUE, hash TEXT UNIQUE, url_group TEXT, tokens INTEGER, file TEXT, row INTEGER, split TEXT, priority TEXT, selected INTEGER DEFAULT 0)")
    total = 0
    for name in files:
        offset = 0
        for batch in pq.ParquetFile(args.root / name).iter_batches(batch_size=4096, columns=["id", "normalized_sha256", "url_group", "llama3_tokens"]):
            for doc in batch.to_pylist():
                group = doc["url_group"]
                is_val = int(hashlib.sha256(("validation-v1:" + group).encode()).hexdigest()[:16], 16) % plan["validation_hash_modulus"] < plan["validation_hash_cutoff"]
                priority = hashlib.sha256(("sample-v1:" + doc["normalized_sha256"]).encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO docs(doc_id,hash,url_group,tokens,file,row,split,priority) VALUES(?,?,?,?,?,?,?,?)",
                           (doc["id"], doc["normalized_sha256"], group, doc["llama3_tokens"], name, offset,
                            "validation" if is_val else "train", priority))
                offset += 1
                total += 1
        db.commit()
    db.execute("CREATE INDEX token_index ON docs(tokens)")
    thresholds = []
    chosen = None
    for length in THRESHOLDS:
        counts = {split: {"documents": n, "tokens": t or 0} for split, n, t in db.execute(
            "SELECT split,COUNT(*),SUM(tokens) FROM docs WHERE tokens>=? GROUP BY split", (length,))}
        thresholds.append({"minimum_llama3_tokens": length, "splits": counts})
        if chosen is None and counts.get("train", {}).get("tokens", 0) >= plan["train_token_target"] and counts.get("validation", {}).get("tokens", 0) >= plan["validation_minimum_tokens"]:
            chosen = length
    if chosen is None:
        write_json(args.root / "insufficient_supply.json", {"thresholds": thresholds, "input_documents": total})
        db.close()
        raise ValueError("Insufficient deduplicated supply; extend extraction before selecting")
    tokens = count = 0
    # Whole documents: the last selected document can exceed the target slightly.
    for rowid, length in db.execute("SELECT rowid,tokens FROM docs WHERE split='train' AND tokens>=? ORDER BY priority", (chosen,)).fetchall():
        db.execute("UPDATE docs SET selected=1 WHERE rowid=?", (rowid,))
        tokens += length
        count += 1
        if tokens >= plan["train_token_target"]:
            break
    # Keep all eligible held-out URL groups from the candidate pool.
    db.execute("UPDATE docs SET selected=1 WHERE split='validation' AND tokens>=?", (chosen,))
    db.execute("CREATE INDEX selected_file_index ON docs(split,selected,file,row)")
    db.commit()
    counts = {s: {"documents": n, "tokens": t, "minimum": lo, "maximum": hi} for s, n, t, lo, hi in db.execute(
        "SELECT split,COUNT(*),SUM(tokens),MIN(tokens),MAX(tokens) FROM docs WHERE selected=1 GROUP BY split")}
    duplicates = total - db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    overlap = db.execute("SELECT COUNT(*) FROM (SELECT url_group FROM docs WHERE selected=1 GROUP BY url_group HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
    if overlap:
        raise ValueError("Validation URL group leakage")
    db.close()
    copied_database = db_path.with_suffix(".partial.sqlite")
    shutil.copyfile(temporary, copied_database)
    copied_database.replace(db_path)
    local_database.cleanup()
    write_json(args.root / "selection.json", {"plan_sha256": plan["plan_sha256"],
        "minimum_llama3_tokens": chosen, "train_target": plan["train_token_target"], "splits": counts,
        "input_documents": total, "duplicate_id_or_normalized_text_removed": duplicates,
        "url_group_overlap": overlap, "thresholds": thresholds, "selection_sqlite_sha256": sha256(db_path),
        "split_policy": "Remove duplicate document IDs. Normalize text with Unicode NFC and collapse whitespace. Remove duplicate normalized text. Assign complete canonical URL groups with a fixed hash. This process does not detect near-duplicate text across unrelated URLs.",
        "selection_policy": "Test minimum lengths in descending order within the candidate pool. Select the highest tested threshold with enough unique tokens for both splits. Select complete training documents with a fixed hash."})
    print(json.dumps({"event": "selected", "minimum_llama3_tokens": chosen, "splits": counts}), flush=True)


def plan_output_tasks(db, source, destination, selection, shard_tokens=128_000_000):
    """Partition selected rows into deterministic, independent output shards."""
    tasks = []
    for split in ["train", "validation"]:
        shard = 0
        task = None
        for name, row, tokens in db.execute(
                "SELECT file,row,tokens FROM docs WHERE selected=1 AND split=? ORDER BY file,row", (split,)):
            if task is None:
                task = {"source": str(source), "destination": str(destination), "split": split,
                        "shard": shard, "minimum": selection["minimum_llama3_tokens"],
                        "files": {}, "tokens": 0, "documents": 0}
            task["files"].setdefault(name, []).append(row)
            task["tokens"] += tokens
            task["documents"] += 1
            if task["tokens"] >= shard_tokens:
                tasks.append(task)
                task = None
                shard += 1
        if task is not None:
            tasks.append(task)
    return tasks


def selected_candidate_batches(path, rows):
    """Decode only row groups that contain selected candidate rows."""
    source = pq.ParquetFile(path)
    rows = sorted(rows)
    offset = seen = 0
    for group in range(source.metadata.num_row_groups):
        size = source.metadata.row_group(group).num_rows
        first = bisect_left(rows, offset)
        last = bisect_left(rows, offset + size, first)
        if last > first:
            indices = pa.array([row - offset for row in rows[first:last]], type=pa.int64())
            selected = source.read_row_group(group, use_threads=False).take(indices)
            yield from selected.to_batches(max_chunksize=32)
            seen += last - first
        offset += size
    if seen != len(rows):
        raise ValueError("Selected candidate rows are outside the source file")


def materialize_shard(task):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source, destination = Path(task["source"]), Path(task["destination"])
    schema = pa.schema([field for field in candidate_schema() if field.name != "input_ids"])
    base = f"{task['split']}-{task['shard']:05d}"
    text_path = destination / "data" / f"{base}.parquet"
    token_path = destination / "tokens" / f"{base}.bin"
    index_path = destination / "tokens" / f"{base}.index.npy"
    cached = task.get("cached_output")
    if cached and cached["tokens"] == task["tokens"] and cached["documents"] == task["documents"]:
        paths = {"text": text_path, "token_ids": token_path, "index": index_path}
        matches = all(cached[kind]["path"] == str(path.relative_to(destination)) and path.is_file()
                      for kind, path in paths.items())
        if matches and all(paths[kind].stat().st_size == cached[kind]["bytes"] for kind in ["text", "token_ids"]):
            if sha256(index_path) == cached["index"]["sha256"]:
                # The following verify stage checks all saved full-file hashes.
                print(json.dumps({"event": "shard_reused", "path": cached["text"]["path"]}), flush=True)
                return cached
    offsets, pending_texts = [], []
    total = 0
    with pq.ParquetWriter(text_path, schema, compression="zstd", use_dictionary=False) as writer, token_path.open("wb") as stream:
        for name, rows in task["files"].items():
            for selected_batch in selected_candidate_batches(source / name, rows):
                token_lists = selected_batch.column("input_ids")
                token_offsets = token_lists.offsets.to_numpy(zero_copy_only=True)
                token_values = token_lists.values.to_numpy(zero_copy_only=True)
                metadata = selected_batch.select([key for key in selected_batch.schema.names
                                                  if key != "input_ids"]).to_pylist()
                for doc_index, doc in enumerate(metadata):
                    ids = np.asarray(token_values[token_offsets[doc_index]:token_offsets[doc_index + 1]], dtype="<u4")
                    if len(ids) != doc["llama3_tokens"] or len(ids) < task["minimum"]:
                        raise ValueError("Invalid selected token length")
                    offsets.append((total, len(ids)))
                    stream.write(ids.tobytes())
                    pending_texts.append(doc)
                    if len(pending_texts) == 32:
                        writer.write_table(pa.Table.from_pylist(pending_texts, schema=schema), row_group_size=32)
                        pending_texts = []
                    total += len(ids)
        if pending_texts:
            writer.write_table(pa.Table.from_pylist(pending_texts, schema=schema), row_group_size=32)
    if total != task["tokens"] or len(offsets) != task["documents"]:
        raise ValueError("Output shard does not match its row plan")
    np.save(index_path, np.asarray(offsets, dtype=np.uint64).reshape(-1, 2))
    return {"split": task["split"], "documents": len(offsets), "tokens": total,
            "text": {"path": str(text_path.relative_to(destination)), "sha256": sha256(text_path), "bytes": text_path.stat().st_size},
            "token_ids": {"path": str(token_path.relative_to(destination)), "sha256": sha256(token_path), "bytes": token_path.stat().st_size},
            "index": {"path": str(index_path.relative_to(destination)), "sha256": sha256(index_path)}}


def materialize_stage(args):
    plan = load_plan(args.root)
    selection = json.loads((args.root / "selection.json").read_text())
    if selection["plan_sha256"] != plan["plan_sha256"] or sha256(args.root / "selection.sqlite") != selection["selection_sqlite_sha256"]:
        raise ValueError("Selection identity mismatch")
    destination = args.output
    if (destination / "manifest.json").exists():
        cached = json.loads((destination / "manifest.json").read_text())
        if not cached.get("complete") or cached["selection"] != selection:
            raise ValueError("Final artifact belongs to another selection")
        print(json.dumps({"event": "materialization_reused", "path": str(destination)}), flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    build_identity = {"plan_sha256": plan["plan_sha256"], "selection_sha256": identity(selection)}
    identity_path = destination / "build_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != build_identity:
        raise ValueError("Destination contains a different build")
    write_json(identity_path, build_identity)
    (destination / "data").mkdir(exist_ok=True)
    (destination / "tokens").mkdir(exist_ok=True)
    db = sqlite3.connect(f"file:{args.root / 'selection.sqlite'}?mode=ro", uri=True)
    tasks = plan_output_tasks(db, args.root, destination, selection)
    db.close()
    progress_path = destination / "materialize_progress.json"
    cached_outputs = {entry["text"]["path"]: entry for entry in json.loads(progress_path.read_text())["files"]} if progress_path.exists() else {}
    checkpoints = dict(cached_outputs)
    for task in tasks:
        task["cached_output"] = cached_outputs.get(f"data/{task['split']}-{task['shard']:05d}.parquet")
    outputs = []
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 32, len(tasks))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(materialize_shard, tasks):
            outputs.append(result)
            print(json.dumps({"event": "materialized", "split": result["split"],
                              "path": result["text"]["path"], "tokens": result["tokens"]}), flush=True)
            checkpoints[result["text"]["path"]] = result
            write_json(progress_path, {"files": [checkpoints[key] for key in sorted(checkpoints)], "complete": False})
    for split in ["train", "validation"]:
        if sum(e["tokens"] for e in outputs if e["split"] == split) != selection["splits"][split]["tokens"]:
            raise ValueError("Materialized token budget disagrees with selection")
    manifest = {"schema_version": 1, "source": {k:plan[k] for k in ["repo_id", "revision", "config"]},
        "selection": selection, "tokenizer": plan["tokenizer"], "tokenizer_sha256": plan["tokenizer_sha256"],
        "implementation": plan.get("implementation"), "materialization_builder_sha256": sha256(Path(__file__)),
        "candidate_sources": plan.get("candidate_sources"), "source_candidate_note": plan["source_candidate_note"],
        "candidate_snapshot_sha256": identity(plan["candidate_snapshot"]) if plan.get("candidate_snapshot") else None,
        "encoding": plan["encoding"], "token_dtype": "uint32 little endian", "token_specials": "None stored; reader adds BOS=128000 and EOS=128001 per document",
        "files": outputs, "complete": True}
    # Upload only original text and metadata; token caches remain local.
    upload_manifest = {**manifest, "files": [{"split": e["split"], "documents": e["documents"], "tokens": e["tokens"], "text": e["text"],
                                             "expected_token_ids_sha256": e["token_ids"]["sha256"]} for e in outputs]}
    write_json(destination / "dataset_manifest.json", upload_manifest)
    write_json(destination / "selection.json", selection)
    write_json(destination / "tokenizer_provenance.json", plan["tokenizer"])
    card = "---\nlanguage:\n- en\nlicense: odc-by\ntask_categories:\n- text-generation\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train-*.parquet\n  - split: validation\n    path: data/validation-*.parquet\n---\n\n# FineWeb-Edu long documents\n\n"
    card += f"This private dataset retains complete `text` records from `{plan['repo_id']}` at revision `{plan['revision']}`.\n\n"
    card += f"Each document contains at least {selection['minimum_llama3_tokens']:,} original Llama 3 tokens. This count excludes special tokens. Each text record remains complete and separate.\n\n"
    card += f"Train: {selection['splits']['train']['tokens']:,} tokens. Validation: {selection['splits']['validation']['tokens']:,} tokens.\n\n"
    card += selection["split_policy"] + "\n\n" + selection["selection_policy"] + "\n\n" + plan["source_candidate_note"] + "\n\n"
    card += "The shared Parquet files contain original text, source URLs, source positions, hashes, and two length fields. A separate local cache contains token IDs. Use the tokenizer version in `tokenizer_provenance.json` to reproduce counts. A complete dataset record can still omit parts of its source webpage or book. Formulas and code formatting can also be incomplete. This pipeline adds no domain classifier or educational-quality classifier.\n\n"
    card += "Source: [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). The source dataset uses ODC-By 1.0. Common Crawl terms also apply. Preserve source attribution. Check the rights for underlying content before downstream use.\n"
    card += "\n## Read the shared text\n\n```python\nfrom datasets import load_dataset\nds = load_dataset('long-sequence-length-dataset/fineweb-edu-long-30b', split='train', streaming=True, token=True)\nrow = next(iter(ds))\nprint(row['llama3_tokens'])\n```\n\nPin a repository commit for reproducible experiments. Organization members use their own read tokens.\n"
    card += "\n## Rebuild the local token cache\n\nDownload the dataset. Download the tokenizer version in `tokenizer_provenance.json`. Access to the original Meta tokenizer may require license acceptance on Hugging Face. Run `python rebuild_long_token_cache.py --dataset-dir /path/to/dataset --tokenizer-dir /path/to/tokenizer`. Install numpy, pyarrow, and tokenizers first. The cache uses about 120 GB for 30B tokens, plus text files. It does not include special tokens or fixed training windows.\n"
    (destination / "README.md").write_text(card)
    (destination / "rebuild_long_token_cache.py").write_bytes((ROOT / "scripts/rebuild_long_token_cache.py").read_bytes())
    import yaml
    training = yaml.safe_load((ROOT / "legacy/configs/examples/llama3_1b_long_documents_smoke.yaml").read_text())
    training["fixed"].update({"training.seq_len": selection["minimum_llama3_tokens"],
                              "validator.eval_seq_len": selection["minimum_llama3_tokens"],
                              "dataloader.dataset_path": str(destination),
                              "validator.dataloader.dataset_path": str(destination)})
    (destination / "training_smoke.yaml").write_text(yaml.safe_dump(training, sort_keys=False))
    write_json(destination / "manifest.json", manifest)
    print(json.dumps({"event": "artifact_complete", "path": str(destination)}), flush=True)


def verify_shard(task):
    root, entry, minimum, vocab_size = task
    root = Path(root)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    for kind in ["text", "token_ids", "index"]:
        if sha256(root / entry[kind]["path"]) != entry[kind]["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {entry[kind]['path']}")
    index = np.load(root / entry["index"]["path"], mmap_mode="r")
    tokens = np.memmap(root / entry["token_ids"]["path"], mode="r", dtype="<u4")
    if len(index) != entry["documents"] or len(tokens) != entry["tokens"]:
        raise ValueError("Token cache shape mismatch")
    tokenizer = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    row = expected_offset = checks = 0
    identities = []
    for batch in pq.ParquetFile(root / entry["text"]["path"]).iter_batches(batch_size=128, use_threads=False):
        for doc in batch.to_pylist():
            offset, length = map(int, index[row])
            ids = tokens[offset:offset + length]
            if offset != expected_offset or length != doc["llama3_tokens"] or length < minimum:
                raise ValueError("Token boundary or minimum length mismatch")
            if len(ids) != length or int(ids.max()) >= vocab_size:
                raise ValueError("Token ID outside tokenizer vocabulary")
            if hashlib.sha256(doc["text"].encode()).hexdigest() != doc["text_sha256"] or normalized_hash(doc["text"]) != doc["normalized_sha256"]:
                raise ValueError("Text content checksum mismatch")
            identities.append((doc["id"], doc["normalized_sha256"], doc["url_group"], entry["split"]))
            if row in {0, len(index) - 1}:
                expected = tokenizer.encode(doc["text"], add_special_tokens=False).ids
                if not np.array_equal(ids, expected):
                    raise ValueError("Tokenization mismatch")
                checks += 1
            expected_offset += length
            row += 1
    if row != len(index) or expected_offset != len(tokens):
        raise ValueError("Text and token cache row mismatch")
    return entry["split"], row, expected_offset, checks, identities


def verify_stage(args):
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    totals = {s: {"documents": 0, "tokens": 0} for s in ["train", "validation"]}
    local_database = tempfile.TemporaryDirectory(prefix="fineweb-verify-", dir=os.environ.get("TMPDIR"))
    database_path = Path(local_database.name) / "verification.sqlite"
    db = sqlite3.connect(database_path)
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA cache_size=-1048576")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("DROP TABLE IF EXISTS seen")
    db.execute("CREATE TABLE seen (id TEXT UNIQUE, hash TEXT UNIQUE, url_group TEXT, split TEXT)")
    if sha256(TOKENIZER / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Verification tokenizer differs from the dataset tokenizer")
    encoded_checks = 0
    tasks = [(str(root), entry, manifest["selection"]["minimum_llama3_tokens"],
              manifest["tokenizer"]["vocab_size"]) for entry in manifest["files"]]
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 32, len(tasks))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for split, count, tokens, checks, identities in pool.map(verify_shard, tasks):
            db.executemany("INSERT INTO seen VALUES(?,?,?,?)", identities)
            db.commit()
            totals[split]["documents"] += count
            totals[split]["tokens"] += tokens
            encoded_checks += checks
            write_json(root / "verify_progress.json", {"totals": totals, "complete": False})
    overlap = db.execute("SELECT COUNT(*) FROM (SELECT url_group FROM seen GROUP BY url_group HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
    db.close()
    shutil.copyfile(database_path, root / "verification.sqlite")
    local_database.cleanup()
    if overlap:
        raise ValueError("Validation URL group leakage")
    for split, counts in totals.items():
        if any(counts[k] != manifest["selection"]["splits"][split][k] for k in counts):
            raise ValueError("Final split counts mismatch")
    result = {"passed": True, "totals": totals, "url_group_overlap": overlap,
              "retokenized_documents": encoded_checks, "manifest_sha256": sha256(root / "manifest.json"),
              "all_text_hashes_and_token_bounds_checked": True}
    if (root / "dataset_manifest.json").exists():
        result["dataset_manifest_sha256"] = sha256(root / "dataset_manifest.json")
    write_json(root / "verification.json", result)
    print(json.dumps(result), flush=True)


def upload_stage(args):
    from huggingface_hub import HfApi, get_token
    if not args.repo_id or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo_id):
        raise ValueError("Supply an explicit organization/repository ID")
    verification = json.loads((args.output / "verification.json").read_text())
    if not verification["passed"] or sha256(args.output / "manifest.json") != verification["manifest_sha256"]:
        raise ValueError("Local artifact must pass verification before upload")
    smoke = json.loads((args.output / "training_smoke.json").read_text())
    if not smoke["passed"] or smoke["manifest_sha256"] != verification["manifest_sha256"]:
        raise ValueError("The final artifact must pass the training smoke test before upload")
    if not get_token():
        raise ValueError("No local HF credential. Log in using hf auth login; do not put tokens in scripts.")
    api = HfApi()
    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    if role == "read":
        raise ValueError("The logged-in token is read-only")
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(args.repo_id)
    if not info.private:
        raise ValueError("Refusing to upload into a public repository")
    api.auth_check(args.repo_id, repo_type="dataset", write=True)
    api.upload_large_folder(args.repo_id, folder_path=args.output, repo_type="dataset", private=True,
        allow_patterns=["data/*.parquet", "README.md", "dataset_manifest.json", "selection.json",
                        "tokenizer_provenance.json", "verification.json", "training_smoke.json", "rebuild_long_token_cache.py",
                        "length_statistics.json"], num_workers=8)
    info = api.dataset_info(args.repo_id, files_metadata=True)
    expected = json.loads((args.output / "dataset_manifest.json").read_text())
    remote = {f.rfilename: f for f in info.siblings}
    for entry in expected["files"]:
        item = entry["text"]
        f = remote[item["path"]]
        if f.size != item["bytes"] or not f.lfs or f.lfs.sha256 != item["sha256"]:
            raise ValueError("Uploaded file checksum mismatch")
    if not info.private:
        raise ValueError("Remote visibility changed")
    from datasets import load_dataset
    remote_examples = {}
    for split in ["train", "validation"]:
        dataset = load_dataset(args.repo_id, split=split, revision=info.sha, streaming=True, token=True)
        doc = next(iter(dataset))
        if doc["llama3_tokens"] < expected["selection"]["minimum_llama3_tokens"] or hashlib.sha256(doc["text"].encode()).hexdigest() != doc["text_sha256"]:
            raise ValueError("Remote dataset loading check failed")
        remote_examples[split] = {"id": doc["id"], "llama3_tokens": doc["llama3_tokens"]}
    write_json(args.output / "upload_receipt.json", {"repo_id": args.repo_id, "revision": info.sha,
        "private": info.private, "verified_text_files": len(expected["files"]), "remote_load_examples": remote_examples,
        "manifest_sha256": verification["manifest_sha256"]})
    print(json.dumps({"event": "uploaded", "repo_id": args.repo_id, "revision": info.sha, "private": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "extract", "combine", "select", "materialize", "verify", "upload"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--candidate-tokens", type=int, default=60_000_000_000)
    parser.add_argument("--train-tokens", type=int, default=30_000_000_000)
    parser.add_argument("--repo-id")
    parser.add_argument("--minimum", type=int, choices=[8192, 12288, 16384], default=16384)
    parser.add_argument("--source-maximum", type=int)
    parser.add_argument("--candidate-root", type=Path, action="append")
    parser.add_argument("--snapshot", action="store_true",
                        help="Combine only completed candidate files, even while extraction continues")
    parser.add_argument("--exhaust-source", action="store_true",
                        help="Resume extraction through all assigned source files, beyond the candidate quota")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.output:
        args.output = args.output.resolve()
    if args.stage in {"materialize", "verify", "upload"} and args.output is None:
        parser.error("--output is required")
    if args.ranks < 1 or args.candidate_tokens < 1 or args.train_tokens < 1:
        parser.error("Invalid rank count or token budgets")
    if args.source_maximum is not None and args.source_maximum <= args.minimum:
        parser.error("The exclusive source maximum must exceed the minimum")
    globals()[args.stage + "_stage"](args)


if __name__ == "__main__":
    main()
