"""Bounded CPU/I/O benchmark; its selected rows are not a distribution sample."""

import argparse
import json
import os
import resource
import socket
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--text-rows-per-shard", type=int, default=1000)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this benchmark inside a Slurm CPU allocation.")
    if args.shards < 1 or args.text_rows_per_shard < 1:
        parser.error("Sample sizes must be positive.")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output already exists; use a new output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    pa.set_io_thread_count(2)
    started = time.perf_counter()
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "scope": "Throughput benchmark on evenly spaced files, prefix texts; not a representative distribution sample.",
        "shards": [],
    }

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(output)

    def progress(event, **fields):
        print(json.dumps({"event": event, **fields}), flush=True)
        checkpoint()

    checkpoint()
    api = HfApi(token=False)
    info = api.dataset_info(report["dataset"])
    report["revision"] = info.sha
    files = sorted(
        (f for f in api.list_repo_tree(report["dataset"], "sample/100BT", repo_type="dataset", revision=info.sha)
         if f.path.endswith(".parquet")), key=lambda f: f.path
    )
    report["source_files"] = len(files)
    report["source_parquet_bytes"] = sum(f.size for f in files)
    report["source_file_sizes"] = {f.path: f.size for f in files}
    try:
        url = "https://datasets-server.huggingface.co/size?" + urllib.parse.urlencode(
            {"dataset": report["dataset"], "config": report["config"]}
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            report["viewer_size_response"] = json.load(response)
    except Exception as exc:
        report["viewer_size_error"] = type(exc).__name__
    selected = sorted(set(np.linspace(0, len(files) - 1, min(args.shards, len(files)), dtype=int).tolist()))
    fs = HfFileSystem(token=False)
    tokenizers = {
        name: Tokenizer.from_file(str(root / "assets/hf" / folder / "tokenizer.json"))
        for name, folder in (("llama2", "llama-tokenizer"), ("llama3", "llama3-tokenizer"))
    }
    # Match the intended measurement: full raw text, no padding/truncation/special tokens.
    for tokenizer in tokenizers.values():
        tokenizer.no_truncation()
        tokenizer.no_padding()
    report["tokenizers"] = {
        name: {"vocab_size": tok.get_vocab_size(), "add_special_tokens": False}
        for name, tok in tokenizers.items()
    }
    progress("source_ready", revision=info.sha, files=len(files))
    for index in selected:
        entry = files[index]
        record = {"path": entry.path, "file_bytes": entry.size, "tokenization": {}}
        report["shards"].append(record)
        remote = f"datasets/{report['dataset']}@{info.sha}/{entry.path}"
        stage = time.perf_counter()
        with fs.open(remote, "rb", block_size=1024 * 1024, cache_type="none") as stream:
            parquet = pq.ParquetFile(stream, pre_buffer=False)
            record["rows_in_shard"] = parquet.metadata.num_rows
            record["row_groups"] = parquet.metadata.num_row_groups
            column_index = parquet.schema_arrow.get_field_index("token_count")
            record["token_count_compressed_bytes"] = sum(
                parquet.metadata.row_group(i).column(column_index).total_compressed_size
                for i in range(parquet.metadata.num_row_groups)
            )
            rows = total_tokens = 0
            thresholds = (4096, 8192, 16384, 32768)
            tail_docs = {str(t): 0 for t in thresholds}
            tail_tokens = {str(t): 0 for t in thresholds}
            for batch in parquet.iter_batches(batch_size=65536, columns=["token_count"], use_threads=False):
                values = batch.column(0).to_numpy(zero_copy_only=False)
                if batch.column(0).null_count or np.any(values < 0):
                    raise ValueError("Unexpected missing or negative length.")
                rows += len(values)
                total_tokens += int(values.sum())
                for threshold in thresholds:
                    eligible = values >= threshold
                    tail_docs[str(threshold)] += int(eligible.sum())
                    tail_tokens[str(threshold)] += int(values[eligible].sum())
            record.update(length_scan_seconds=time.perf_counter() - stage,
                          scanned_rows=rows, source_tokens=total_tokens,
                          tail_documents=tail_docs, tail_source_tokens=tail_tokens)
        progress("length_scan_complete", path=entry.path, rows=rows, seconds=record["length_scan_seconds"])
        stage = time.perf_counter()
        texts = []
        with fs.open(remote, "rb", block_size=1024 * 1024, cache_type="none") as stream:
            parquet = pq.ParquetFile(stream, pre_buffer=False)
            for batch in parquet.iter_batches(batch_size=64, columns=["text"], use_threads=False):
                texts.extend(batch.column(0).to_pylist()[:args.text_rows_per_shard - len(texts)])
                if len(texts) >= args.text_rows_per_shard:
                    break
        record["text_read_seconds"] = time.perf_counter() - stage
        record["text_documents"] = len(texts)
        record["text_utf8_bytes"] = sum(len(s.encode("utf-8")) for s in texts)
        for name, tokenizer in tokenizers.items():
            tokenizer.encode("Warm up the tokenizer.", add_special_tokens=False)
            stage = time.perf_counter()
            total = 0
            for offset in range(0, len(texts), 32):
                encodings = tokenizer.encode_batch(texts[offset:offset + 32], add_special_tokens=False)
                total += sum(len(encoded.ids) for encoded in encodings)
                del encodings
            elapsed = time.perf_counter() - stage
            record["tokenization"][name] = {"seconds": elapsed, "tokens": total,
                "tokens_per_second": total / elapsed, "documents_per_second": len(texts) / elapsed}
            progress("tokenizer_complete", name=name, path=entry.path, seconds=elapsed, tokens=total)
        del texts
    report["completed"] = True
    progress("complete", peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


if __name__ == "__main__":
    main()
