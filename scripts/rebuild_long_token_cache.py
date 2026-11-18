"""Rebuild local uint32 token caches from the shared text-only dataset.

Dependencies: numpy, pyarrow, tokenizers. Download the dataset and the pinned
tokenizer first. This script does not need a network connection or access token.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _tokenizer(path):
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def reencode_shard(root, tokenizer_path, entry, *, resume=False):
    """Encode one complete shard. Check source and output hashes before reuse."""
    started = time.monotonic()
    text_path = root / entry["text"]["path"]
    if sha256(text_path) != entry["text"]["sha256"]:
        raise ValueError(f"Text Parquet checksum mismatch: {text_path}")
    folder = root / "tokens"
    base = text_path.stem
    binary = folder / f"{base}.bin"
    index_path = folder / f"{base}.index.npy"
    # An index without an independent reference hash cannot pass this shortcut.
    reused = bool(resume and entry.get("expected_index_sha256")
                  and binary.is_file() and index_path.is_file()
                  and sha256(binary) == entry["expected_token_ids_sha256"]
                  and sha256(index_path) == entry["expected_index_sha256"])
    if not reused:
        tokenizer = _tokenizer(str(tokenizer_path / "tokenizer.json"))
        temporary = binary.with_suffix(".partial.bin")
        temporary_index = index_path.with_suffix(".partial.npy")
        offsets = []
        total = 0
        with temporary.open("wb") as stream:
            for batch in pq.ParquetFile(text_path).iter_batches(
                    batch_size=32, columns=["text", "text_sha256", "llama3_tokens"]):
                docs = batch.to_pylist()
                encodings = tokenizer.encode_batch([d["text"] for d in docs], add_special_tokens=False)
                for doc, encoded in zip(docs, encodings):
                    if hashlib.sha256(doc["text"].encode()).hexdigest() != doc["text_sha256"]:
                        raise ValueError("Document text hash mismatch")
                    ids = np.asarray(encoded.ids, dtype="<u4")
                    if len(ids) != doc["llama3_tokens"]:
                        raise ValueError("Document token count mismatch")
                    stream.write(ids.tobytes())
                    offsets.append((total, len(ids)))
                    total += len(ids)
        if total != entry["tokens"] or len(offsets) != entry["documents"]:
            raise ValueError("Token shard budget mismatch")
        if sha256(temporary) != entry["expected_token_ids_sha256"]:
            raise ValueError("Rebuilt token IDs do not match the original token cache")
        np.save(temporary_index, np.asarray(offsets, dtype=np.uint64))
        if entry.get("expected_index_sha256") and sha256(temporary_index) != entry["expected_index_sha256"]:
            raise ValueError("Rebuilt document index differs from the original index")
        temporary.replace(binary)
        temporary_index.replace(index_path)
    metadata = {k: v for k, v in entry.items()
                if k not in ("expected_token_ids_sha256", "expected_index_sha256")}
    result = {**metadata,
              "token_ids": {"path": str(binary.relative_to(root)), "sha256": sha256(binary), "bytes": binary.stat().st_size},
              "index": {"path": str(index_path.relative_to(root)), "sha256": sha256(index_path)}}
    progress = {"shard": base, "tokens": entry["tokens"], "reused_checked_cache": reused,
                "elapsed_seconds": time.monotonic() - started}
    return result, progress


def _encode_job(job):
    root, tokenizer_path, entry, resume = job
    return reencode_shard(root, tokenizer_path, entry, resume=resume)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rebuild(root, tokenizer_path, *, workers=1, resume=False, limit_shards=None):
    """Rebuild all shards, or benchmark a prefix without publishing completion."""
    started = time.monotonic()
    root, tokenizer_path = Path(root), Path(tokenizer_path)
    if workers < 1 or (limit_shards is not None and limit_shards < 1):
        raise ValueError("Worker and shard limits must be positive")
    shared = json.loads((root / "dataset_manifest.json").read_text())
    receipt = json.loads((root / "verification.json").read_text())
    if not receipt["passed"] or receipt.get("dataset_manifest_sha256") != sha256(root / "dataset_manifest.json"):
        raise ValueError("Shared text manifest does not match its verification receipt")
    if sha256(tokenizer_path / "tokenizer.json") != shared["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match the dataset's pinned tokenizer")
    for split, entry in shared.get("document_orders", {}).items():
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Document order checksum mismatch: {split}")
    folder = root / "tokens"
    if folder.is_symlink():
        raise ValueError("Refusing to rebuild into a linked token cache; use a separate downloaded copy")
    folder.mkdir(exist_ok=True)
    names = [Path(entry["text"]["path"]).stem for entry in shared["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Shard names must be unique")
    entries = shared["files"][:limit_shards]
    jobs = [(root, tokenizer_path, entry, resume) for entry in entries]
    files = []
    progress = {"complete": False, "workers": workers, "requested_shards": len(entries),
                "total_shards": len(shared["files"]), "shards": []}
    _write_json(root / "rebuild_progress.json", progress)

    def collect(results):
        for metadata, result in results:
            files.append(metadata)
            progress["shards"].append(result)
            progress["elapsed_seconds"] = time.monotonic() - started
            _write_json(root / "rebuild_progress.json", progress)
            print(json.dumps(result), flush=True)

    if workers == 1:
        collect(map(_encode_job, jobs))
    else:
        # Spawn prevents a child from inheriting a tokenizer thread pool.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            collect(pool.map(_encode_job, jobs))
    if limit_shards is not None:
        progress["benchmark_only"] = True
        _write_json(root / "rebuild_progress.json", progress)
        return progress
    manifest = {**shared, "files": files, "complete": True}
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    if hashlib.sha256(manifest_bytes).hexdigest() != receipt["manifest_sha256"]:
        raise ValueError("Rebuilt dataset manifest differs from the shared identity")
    _write_json(root / "manifest.json", manifest)
    # The shared verification receipt binds the text artifact and split isolation.
    receipt.setdefault("shared_verification_manifest_sha256", receipt["manifest_sha256"])
    receipt["manifest_sha256"] = sha256(root / "manifest.json")
    receipt["all_token_cache_documents_reencoded"] = True
    _write_json(root / "verification.json", receipt)
    progress["complete"] = True
    progress["elapsed_seconds"] = time.monotonic() - started
    progress["manifest_sha256"] = sha256(root / "manifest.json")
    _write_json(root / "rebuild_progress.json", progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Parallel shard processes. Set RAYON_NUM_THREADS for each process.")
    parser.add_argument("--resume", action="store_true", help="Reuse only shards that pass all source, token, and index hash checks.")
    parser.add_argument("--limit-shards", type=int, help="Benchmark this many shards. Do not publish a complete cache.")
    args = parser.parse_args()
    rebuild(args.dataset_dir.resolve(), args.tokenizer_dir.resolve(), workers=args.workers,
            resume=args.resume, limit_shards=args.limit_shards)


if __name__ == "__main__":
    main()
