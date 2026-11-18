"""Select source documents near specified lengths for manual examination."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "outputs/fineweb_edu_profile/full-100bt"
OUTPUT = ROOT / "outputs/fineweb_edu_profile/document_examples"
TARGETS = (2096, 8192, 16384, 32768)
POOL_SIZES = {2096: 1024, 8192: 256, 16384: 256, 32768: 1024}
# Prefer agency publications for complete text examples. Review each selected source before export.
AGENCIES = ("nasa.gov", "noaa.gov", "weather.gov", "usgs.gov", "nps.gov", "epa.gov",
            "usda.gov", "fs.fed.us", "census.gov", "energy.gov", "nist.gov", "cdc.gov",
            "federalreserve.gov", "fda.gov", "bls.gov", "gao.gov", "gpo.gov", "govinfo.gov", "osha.gov")


def save(name, data):
    path = OUTPUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def open_source(entry, manifest):
    meta = get_hf_file_metadata(hf_hub_url(manifest["repo_id"], entry["path"],
        revision=manifest["revision"], repo_type="dataset"), token=False, timeout=30)
    if meta.commit_hash != manifest["revision"] or meta.size != entry["size"]:
        raise ValueError("Source identity mismatch")
    fs = HTTPFileSystem(client_kwargs={"timeout": aiohttp.ClientTimeout(total=90)}, skip_instance_cache=True)
    return fs.open(meta.location, "rb", size=meta.size, block_size=1024 * 1024, cache_type="none")


def read_rows(entry, requested, manifest, columns):
    for attempt in range(3):
        try:
            result = []
            with open_source(entry, manifest) as stream:
                parquet = pq.ParquetFile(stream, pre_buffer=False)
                starts = np.cumsum([0] + [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)])
                groups = {}
                for sample in requested:
                    row = sample["row"]
                    group = int(np.searchsorted(starts, row, side="right") - 1)
                    groups.setdefault(group, []).append(sample)
                chosen_columns = [c for c in columns if c in parquet.schema_arrow.names]
                for group, samples in groups.items():
                    table = parquet.read_row_group(group, columns=chosen_columns, use_threads=False)
                    for sample in samples:
                        row = table.slice(sample["row"] - int(starts[group]), 1).to_pylist()[0]
                        if row["token_count"] != sample["source_tokens"]:
                            raise ValueError("Document length differs from the saved index")
                        result.append({**sample, **row})
            return result
        except Exception as exc:
            print(json.dumps({"event": "retry", "file": entry["path"], "attempt": attempt + 1,
                              "error_type": type(exc).__name__}), flush=True)
            if attempt == 2:
                raise
            time.sleep(2 + attempt * 2)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this program in a CPU allocation.")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROFILE / "source_manifest.json").read_text())
    rng = np.random.default_rng(20261104)
    pools = {target: [] for target in TARGETS}
    eligible_counts = {target: 0 for target in TARGETS}
    for entry in manifest["files"]:
        values = np.load(PROFILE / "length_index" / (Path(entry["path"]).stem + ".npy"), mmap_mode="r")
        for target in TARGETS:
            low, high = int(np.ceil(target * .98)), int(np.floor(target * 1.02))
            rows = np.flatnonzero((values >= low) & (values <= high))
            eligible_counts[target] += len(rows)
            priorities = rng.random(len(rows))
            keep = np.argsort(priorities)[:POOL_SIZES[target]]
            pools[target].extend({"target": target, "path": entry["path"], "row": int(rows[i]),
                                  "source_tokens": int(values[rows[i]]), "priority": float(priorities[i])} for i in keep)
            pools[target] = sorted(pools[target], key=lambda x: x["priority"])[:POOL_SIZES[target]]
    selection = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                 "config": manifest["config"], "seed": 20261104, "relative_tolerance": .02,
                 "targets": TARGETS, "candidate_pool_sizes": POOL_SIZES, "eligible_documents": eligible_counts,
                 "method": "Random priority sample within each length range. Select agency publications for full text inspection. This is not a representative sample.",
                 "job_id": os.environ["SLURM_JOB_ID"]}
    save("selection.json", selection)
    cached_path = OUTPUT / "candidate_metadata.json"
    cached_metadata = {(d["target"], d["path"], d["row"]): d
                       for d in json.loads(cached_path.read_text())} if cached_path.exists() else {}
    metadata = []
    grouped = {}
    for samples in pools.values():
        for sample in samples:
            identity = (sample["target"], sample["path"], sample["row"])
            if identity in cached_metadata:
                metadata.append(cached_metadata[identity])
            else:
                grouped.setdefault(sample["path"], []).append(sample)
    entries = {e["path"]: e for e in manifest["files"]}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(read_rows, entries[path], samples, manifest,
                   ["id", "url", "dump", "token_count", "score", "int_score"]): path
                   for path, samples in grouped.items()}
        for future in as_completed(futures):
            metadata.extend(future.result())
            print(json.dumps({"event": "metadata", "documents": len(metadata)}), flush=True)
    save("candidate_metadata.json", sorted(metadata, key=lambda d: (d["target"], d["priority"])))
    selected = []
    for target in TARGETS:
        candidates = []
        for doc in metadata:
            if doc["target"] != target:
                continue
            host = (urlsplit(doc["url"]).hostname or "").lower()
            if any(host == domain or host.endswith("." + domain) for domain in AGENCIES):
                candidates.append(doc)
        candidates.sort(key=lambda d: (abs(d["source_tokens"] - target), d["priority"]))
        seen = set()
        for doc in candidates:
            canonical = doc["url"].split("?", 1)[0].rstrip("/")
            if canonical not in seen:
                selected.append(doc)
                seen.add(canonical)
            if len(seen) == 8:
                break
        print(json.dumps({"event": "agency_candidates", "target": target, "candidates": len(candidates), "selected": len(seen)}), flush=True)
    grouped = {}
    for doc in selected:
        grouped.setdefault(doc["path"], []).append(doc)
    documents = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_rows, entries[path], samples, manifest,
                   ["id", "url", "dump", "token_count", "score", "int_score", "text"])
                   for path, samples in grouped.items()]
        for future in as_completed(futures):
            documents.extend(future.result())
    tokenizers = {name: Tokenizer.from_file(str(ROOT / "assets/hf" / folder / "tokenizer.json"))
                  for name, folder in (("llama2", "llama-tokenizer"), ("llama3", "llama3-tokenizer"))}
    for tokenizer in tokenizers.values():
        tokenizer.no_truncation()
        tokenizer.no_padding()
    for doc in documents:
        doc["text_sha256"] = hashlib.sha256(doc["text"].encode()).hexdigest()
        doc["characters"] = len(doc["text"])
        doc["lines"] = len(doc["text"].splitlines())
        doc["words_whitespace"] = len(doc["text"].split())
        doc["tokenizer_lengths_without_special_tokens"] = {
            name: len(tokenizer.encode(doc["text"], add_special_tokens=False).ids)
            for name, tokenizer in tokenizers.items()}
    save("agency_documents.json", sorted(documents, key=lambda d: (d["target"], abs(d["source_tokens"] - d["target"]))))
    save("document_overview.json", [{k: v for k, v in doc.items() if k != "text"} for doc in documents])
    print(json.dumps({"event": "complete", "documents": len(documents)}), flush=True)


if __name__ == "__main__":
    main()
