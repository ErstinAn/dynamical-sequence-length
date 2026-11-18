"""Exact, resumable length-field profiling of a pinned FineWeb-Edu snapshot."""

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import get_hf_file_metadata, hf_hub_url


EDGES = (0, 1, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
THRESHOLDS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_from_counts(counts, q):
    cumulative = np.cumsum(counts)
    n = int(cumulative[-1])
    position = (n - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    a = int(np.searchsorted(cumulative, lo, side="right"))
    b = int(np.searchsorted(cumulative, hi, side="right"))
    return a + (b - a) * (position - lo)


def summarize(counts):
    counts = np.asarray(counts, dtype=np.int64)
    positions = np.arange(len(counts), dtype=np.int64)
    n = int(counts.sum())
    total = int(np.dot(counts, positions))
    nonzero = np.flatnonzero(counts)
    if not n:
        raise ValueError("No documents were counted.")
    buckets = []
    for i, lower in enumerate(EDGES):
        upper = EDGES[i + 1] if i + 1 < len(EDGES) else None
        selected = slice(lower, upper)
        documents = int(counts[selected].sum())
        tokens = int(np.dot(counts[selected], positions[selected]))
        buckets.append({"lower_inclusive": lower, "upper_exclusive": upper,
                        "documents": documents, "document_fraction": documents / n,
                        "source_tokens": tokens, "token_fraction": tokens / total if total else 0})
    tails = []
    for threshold in THRESHOLDS:
        documents = int(counts[threshold:].sum())
        tokens = int(np.dot(counts[threshold:], positions[threshold:]))
        full_windows = int(np.dot(counts, positions // threshold))
        tails.append({"minimum_length": threshold, "documents": documents,
                      "document_fraction": documents / n, "source_tokens": tokens,
                      "token_fraction": tokens / total if total else 0,
                      "full_windows_in_source_token_units": full_windows,
                      "tokens_in_full_windows": full_windows * threshold})
    return {"documents": n, "source_tokens": total,
            "min": int(nonzero[0]), "max": int(nonzero[-1]), "mean": total / n,
            "quantiles_linear": {str(q): quantile_from_counts(counts, q)
                                 for q in (0.5, 0.9, 0.95, 0.99, 0.999)},
            "buckets": buckets, "thresholds": tails}


def self_test():
    # Compare histogram-based aggregates with direct calculations at every bucket boundary.
    values = np.array([0, 1, 255, 256, 511, 512, 1023, 1024, 2047, 2048,
                       4095, 4096, 8191, 8192, 16383, 16384, 32768, 65536, 100000])
    counts = np.bincount(values)
    summary = summarize(counts)
    assert summary["documents"] == len(values)
    assert summary["source_tokens"] == int(values.sum())
    assert sum(b["documents"] for b in summary["buckets"]) == len(values)
    assert sum(b["source_tokens"] for b in summary["buckets"]) == int(values.sum())
    for q, observed in summary["quantiles_linear"].items():
        assert np.isclose(observed, np.quantile(values, float(q)))
    for tail in summary["thresholds"]:
        threshold = tail["minimum_length"]
        assert tail["documents"] == int((values >= threshold).sum())
        assert tail["source_tokens"] == int(values[values >= threshold].sum())
        assert tail["tokens_in_full_windows"] == int((values // threshold).sum()) * threshold
    print(json.dumps({"event": "self_test_passed"}), flush=True)


def scan_file(entry, manifest, output):
    # Default contains many crawl directories with identical file basenames.
    name = (entry["path"].removesuffix(".parquet").replace("/", "__")
            if manifest["config"] == "default" else Path(entry["path"]).stem)
    final = output / "shards" / (name + ".json")
    index = output / "length_index" / (name + ".npy")
    histogram = output / "histograms" / (name + ".npz")
    if final.exists():
        cached = json.loads(final.read_text())
        if (cached["revision"] != manifest["revision"] or cached["source"] != entry
                or sha256(index) != cached["length_index_sha256"]
                or sha256(histogram) != cached["histogram_sha256"]):
            raise ValueError("Cached shard identity/checksum mismatch: " + name)
        return cached
    started = time.perf_counter()
    last_error = None
    for attempt in range(3):
        try:
            meta = get_hf_file_metadata(hf_hub_url(manifest["repo_id"], entry["path"],
                repo_type="dataset", revision=manifest["revision"]), token=False, timeout=30)
            if meta.size != entry["size"] or meta.commit_hash != manifest["revision"]:
                raise ValueError("Remote source identity mismatch")
            # Resolve once, then range-read directly from the file's download server.
            fs = HTTPFileSystem(client_kwargs={"timeout": aiohttp.ClientTimeout(total=90)},
                                skip_instance_cache=True)
            with fs.open(meta.location, "rb", size=meta.size, block_size=1024 * 1024,
                         cache_type="none") as stream:
                parquet = pq.ParquetFile(stream, pre_buffer=False)
                expected_rows = parquet.metadata.num_rows
                temporary = index.with_suffix(".partial.npy")
                lengths = np.lib.format.open_memmap(temporary, mode="w+", dtype="<u4", shape=(expected_rows,))
                counts = np.zeros(1, dtype=np.int64)
                offset = 0
                for batch in parquet.iter_batches(batch_size=16384, columns=["token_count"], use_threads=False):
                    column = batch.column(0)
                    if column.null_count:
                        raise ValueError("Null token_count")
                    values = column.to_numpy(zero_copy_only=False)
                    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
                        raise ValueError("Invalid token_count")
                    # Fail explicitly on unexpected metadata instead of silently dropping records.
                    if len(values) and int(values.max()) > 10_000_000:
                        raise ValueError("Unexpected length over 10 million; inspect before continuing")
                    new_counts = np.bincount(values.astype(np.int64, copy=False))
                    if len(new_counts) > len(counts):
                        counts = np.pad(counts, (0, len(new_counts) - len(counts)))
                    counts[:len(new_counts)] += new_counts
                    lengths[offset:offset + len(values)] = values
                    offset += len(values)
                if offset != expected_rows or int(counts.sum()) != expected_rows:
                    raise ValueError("Source row count mismatch")
                lengths.flush()
                del lengths
                temporary.replace(index)
                summary = summarize(counts)
                col = parquet.schema_arrow.get_field_index("token_count")
                compressed_bytes = sum(parquet.metadata.row_group(i).column(col).total_compressed_size
                                       for i in range(parquet.metadata.num_row_groups))
            np.savez_compressed(histogram, counts=counts)
            result = {"source": entry, "revision": manifest["revision"],
                      "length_index": str(index.relative_to(output)), "length_index_dtype": "uint32",
                      "row_number_base": 0, "length_index_sha256": sha256(index),
                      "histogram": str(histogram.relative_to(output)), "histogram_sha256": sha256(histogram),
                      "length_column_compressed_bytes": compressed_bytes,
                      "elapsed_seconds": time.perf_counter() - started, "attempts": attempt + 1,
                      "statistics": summary}
            write_json(final, result)
            return result
        except Exception as exc:
            last_error = type(exc).__name__
            print(json.dumps({"event": "shard_retry", "file": entry["path"],
                              "attempt": attempt + 1, "error_type": last_error}), flush=True)
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(name + ": " + str(last_error))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run in a Slurm CPU allocation.")
    self_test()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = Path(args.output).resolve()
    manifest = json.loads(Path(args.source_manifest).read_text())
    files = manifest["files"]
    if len({x["path"] for x in files}) != len(files) or args.workers < 1:
        raise ValueError("Duplicate files or invalid worker count")
    for folder in (output, output / "shards", output / "length_index", output / "histograms"):
        folder.mkdir(parents=True, exist_ok=True)
    source_record = output / "source_manifest.json"
    if source_record.exists() and json.loads(source_record.read_text()) != manifest:
        raise ValueError("Existing output belongs to a different manifest")
    write_json(source_record, manifest)
    started = time.perf_counter()
    base = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
            "config": manifest["config"], "length_basis": "source token_count; not Llama 2/3 lengths",
            "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
            "workers": args.workers, "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "expected_files": len(files),
            "scope": "Exact full selected config; all rows retained including zero-length records.",
            "script_sha256": sha256(Path(__file__)),
            "window_count_note": "Arithmetic supply in source token units only; excludes tokenizer changes, special tokens, deduplication and split exclusions."}
    completed = []
    failures = []
    combined = np.zeros(1, dtype=np.int64)

    def save_progress():
        elapsed = time.perf_counter() - started
        stats = summarize(combined) if combined.sum() else None
        result = {**base, "completed_files": len(completed), "failed_files": failures,
                  "elapsed_seconds": elapsed, "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "complete": len(completed) == len(files) and not failures,
                  "partial": len(completed) != len(files), "statistics": stats,
                  "completed_paths": sorted(completed)}
        write_json(output / "progress.json", result)
        return result

    save_progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_file, entry, manifest, output): entry for entry in files}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                record = future.result()
                with np.load(output / record["histogram"]) as saved:
                    counts = saved["counts"]
                if len(counts) > len(combined):
                    combined = np.pad(combined, (0, len(counts) - len(combined)))
                combined[:len(counts)] += counts
                completed.append(entry["path"])
            except Exception as exc:
                failures.append({"path": entry["path"], "error_type": type(exc).__name__})
            report = save_progress()
            print(json.dumps({"event": "progress", "completed_files": len(completed),
                              "failed_files": len(failures), "expected_files": len(files),
                              "documents": report["statistics"]["documents"] if report["statistics"] else 0,
                              "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    if failures:
        raise SystemExit("Incomplete profile; inspect progress.json and rerun to resume.")
    report = save_progress()
    summary = report["statistics"]
    if manifest.get("expected_documents") is not None and summary["documents"] != manifest["expected_documents"]:
        report.update(complete=False, validation_error="Total rows disagree with independent expected count")
        write_json(output / "progress.json", report)
        raise SystemExit(report["validation_error"])
    assert sum(b["documents"] for b in summary["buckets"]) == summary["documents"]
    assert sum(b["source_tokens"] for b in summary["buckets"]) == summary["source_tokens"]
    np.savez_compressed(output / "length_histogram.npz", counts=combined)
    write_csv(output / "length_buckets.csv", summary["buckets"])
    write_csv(output / "length_thresholds.csv", summary["thresholds"])
    write_json(output / "report.json", report)
    print(json.dumps({"event": "complete", "report": str(output / "report.json"),
                      "documents": summary["documents"], "source_tokens": summary["source_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
