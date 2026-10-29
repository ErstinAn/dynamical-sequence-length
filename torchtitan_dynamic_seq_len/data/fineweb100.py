"""Load the fixed FineWeb training split and prepare validation data.

Training excludes the complete validation source file.
Training also excludes copies of evaluated documents by ID and normalized text hash."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from .fineweb100_identity import verify_validation_identity

REPO_ID = "HuggingFaceFW/fineweb"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
CONFIG = "sample-100BT"
VALIDATION_FILE = "sample/100BT/000_00000.parquet"
DEFAULT_TOKEN_BUDGET = 20_971_520
MANIFEST_PATH = Path(__file__).parent / "manifests" / "fineweb_100bt_manifest.json"
DEFAULT_VALIDATION_DIR = Path(__file__).parents[2] / "assets" / "data" / "fineweb100_validation"
NORMALIZATION = "Unicode NFC, then collapse Unicode whitespace with str.split/join"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validation_directory(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("DYNAMIC_SEQ_LEN_VALIDATION_DIR") or DEFAULT_VALIDATION_DIR)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest = json.loads(Path(path or MANIFEST_PATH).read_text())
    expected = {
        f"sample/100BT/{group:03d}_{part:05d}.parquet" for group in range(15) for part in range(10)
    }
    if (
        manifest.get("schema_version"),
        manifest.get("repo_id"),
        manifest.get("revision"),
        manifest.get("config"),
    ) != (1, REPO_ID, REVISION, CONFIG):
        raise ValueError("FineWeb 100BT manifest identity does not match the pinned protocol")
    source_files = manifest["source_files"]
    source_paths = [entry["path"] for entry in source_files]
    train = manifest["training_files"]
    validation = manifest["validation_files"]
    if len(source_paths) != 150 or set(source_paths) != expected:
        raise ValueError("Manifest must contain all 150 unique FineWeb 100BT files")
    if (
        validation != [VALIDATION_FILE]
        or len(train) != 149
        or set(train) != expected - {VALIDATION_FILE}
    ):
        raise ValueError(
            "Training must contain exactly the 149 files excluding validation shard 000"
        )
    if train != sorted(train):
        raise ValueError("Training file order must be frozen in increasing shard order")
    for entry in source_files:
        if (
            not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
        ):
            raise ValueError(f"Missing size or SHA256 for {entry['path']}")
    return manifest


def training_data_files(dataset_path: str, manifest_path: str | Path | None = None) -> list[str]:
    manifest = load_manifest(manifest_path)
    if dataset_path == REPO_ID:
        return [f"hf://datasets/{REPO_ID}@{REVISION}/{path}" for path in manifest["training_files"]]
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Expected {REPO_ID!r} or a complete local snapshot: {root}")
    paths = [root / path for path in manifest["training_files"]]
    sizes = {entry["path"]: entry["size"] for entry in manifest["source_files"]}
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete FineWeb 100BT snapshot ({len(missing)} training shards missing): {missing[:3]}"
        )
    wrong_size = [
        str(path)
        for path, rel in zip(paths, manifest["training_files"])
        if path.stat().st_size != sizes[rel]
    ]
    if wrong_size:
        raise ValueError(
            f"FineWeb 100BT local shard sizes differ from the pinned manifest: {wrong_size[:3]}"
        )
    return [str(path) for path in paths]


def load_evaluation_exclusions(path: str | Path | None = None) -> tuple[set[str], set[str]]:
    """Verify the small exclusion index before training starts (fail closed)."""
    root = validation_directory(path)
    try:
        metadata = json.loads((root / "manifest.json").read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Prepare the frozen FineWeb validation artifact first, or set DYNAMIC_SEQ_LEN_VALIDATION_DIR: {root}"
        ) from error
    source = metadata.get("source", {})
    expected = next(
        entry for entry in load_manifest()["source_files"] if entry["path"] == VALIDATION_FILE
    )
    if (
        source.get("repo_id"),
        source.get("revision"),
        source.get("path"),
        source.get("sha256"),
    ) != (REPO_ID, REVISION, VALIDATION_FILE, expected["sha256"]):
        raise ValueError("Validation exclusion index is from a different dataset/shard")
    verify_validation_identity(metadata)
    index = metadata["evaluation_documents"]
    if (
        index.get("file") != "evaluation_documents.jsonl"
        or index.get("text_normalization") != NORMALIZATION
    ):
        raise ValueError("Unsupported validation exclusion index format")
    index_path = root / index["file"]
    if file_sha256(index_path) != index["sha256"]:
        raise ValueError("Validation document exclusion index checksum mismatch")
    ids, hashes = set(), set()
    with index_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            ids.add(row["id"])
            hashes.add(row["text_sha256"])
    if not ids or len(ids) != index["count"]:
        raise ValueError("Validation document exclusion index is empty or has duplicate IDs")
    return ids, hashes


def load_fineweb100(dataset_path: str):
    """Stream all 149 training shards, never the held-out shard or eval copies."""
    from datasets import load_dataset

    paths = training_data_files(dataset_path)
    ids, hashes = load_evaluation_exclusions()
    dataset = load_dataset("parquet", data_files={"train": paths}, split="train", streaming=True)

    def is_training_document(sample: dict[str, Any]) -> bool:
        return str(sample["id"]) not in ids and normalized_text_hash(sample["text"]) not in hashes

    return dataset.filter(is_training_document)


def register_fineweb100() -> None:
    from torchtitan.hf_datasets import DatasetConfig
    from torchtitan.hf_datasets.text_datasets import DATASETS

    DATASETS.setdefault(
        "fineweb_100bt",
        DatasetConfig(
            path=REPO_ID, loader=load_fineweb100, sample_processor=lambda sample: sample["text"]
        ),
    )


def pack_documents(documents: Iterable[dict[str, Any]], tokenizer: Any, seq_len: int):
    """Create labels and positions within each document.

    The fourth value identifies the documents in each packed sequence.
    Data preparation packs the sequences before DP partitioning."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    inputs, labels, positions, ids = [], [], [], []
    for doc in documents:
        tokens = tokenizer.encode(doc["text"], add_bos=True, add_eos=True)
        length = len(tokens) - 1
        if length <= 0:
            continue
        inputs.extend(tokens[:-1])
        labels.extend(tokens[1:])
        positions.extend(range(length))
        ids.extend([str(doc["id"])] * length)
        while len(inputs) >= seq_len:
            pos = positions[:seq_len]
            offset = pos[0]
            if offset > 0:
                for index, value in enumerate(pos):
                    if value == 0:
                        break
                    pos[index] -= offset
            yield inputs[:seq_len], labels[:seq_len], pos, list(dict.fromkeys(ids[:seq_len]))
            inputs, labels, positions, ids = (
                values[seq_len:] for values in (inputs, labels, positions, ids)
            )


def prepare_validation(
    source_parquet: str | Path,
    output: str | Path,
    tokenizer_path: str | Path,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    seq_len: int = 2048,
    *,
    manifest_path: str | Path | None = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Create a fixed validation artifact from the source file.

    The tokenizer reads enough documents for the token budget and one additional sequence.
    The source token count uses the GPT-2 tokenizer.
    That count is separate from the token count in this artifact."""
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source_parquet, output, tokenizer_path = map(Path, (source_parquet, output, tokenizer_path))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation artifact: {output}")
    if seq_len <= 0 or token_budget <= 0 or token_budget % seq_len:
        raise ValueError("token_budget must be positive and divisible by seq_len")
    manifest = load_manifest(manifest_path)
    source = next(entry for entry in manifest["source_files"] if entry["path"] == VALIDATION_FILE)
    if (
        source_parquet.stat().st_size != source["size"]
        or file_sha256(source_parquet) != source["sha256"]
    ):
        raise ValueError("Validation source parquet does not match pinned size/SHA256")
    if tokenizer is None:
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tokenizer_path))
    tokenizer_files = {
        path.name: file_sha256(path)
        for path in sorted(tokenizer_path.iterdir())
        if path.is_file() and path.suffix in {".json", ".txt", ".model", ".jinja"}
    }
    if not tokenizer_files:
        raise ValueError("Tokenizer assets are empty")
    parquet = pq.ParquetFile(source_parquet)
    if not {"id", "text", "token_count"} <= set(parquet.schema_arrow.names):
        raise ValueError("Validation source lacks FineWeb id/text/token_count columns")
    official_tokens = sum(
        int(pc.sum(batch.column(0)).as_py() or 0)
        for batch in parquet.iter_batches(batch_size=131072, columns=["token_count"])
    )
    if official_tokens <= 0:
        raise ValueError("Validation source has invalid official token_count")

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.prepare.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        count = token_budget // seq_len
        arrays = {
            key: np.lib.format.open_memmap(
                temporary / f"{key}.npy", mode="w+", dtype="<i8", shape=(count, seq_len)
            )
            for key in ("input", "labels", "positions")
        }
        observed_docs: dict[str, str] = {}
        used_ids: dict[str, None] = {}
        prefix = {"documents": 0, "label_tokens": 0}

        class CountingTokenizer:
            def encode(self, *args, **kwargs):
                values = tokenizer.encode(*args, **kwargs)
                prefix["label_tokens"] += max(0, len(values) - 1)
                return values

        def documents():
            for batch in parquet.iter_batches(batch_size=256, columns=["id", "text"]):
                for doc in batch.to_pylist():
                    doc_id = str(doc["id"])
                    text_hash = normalized_text_hash(doc["text"])
                    if doc_id in observed_docs and observed_docs[doc_id] != text_hash:
                        raise ValueError(
                            f"Source contains conflicting text for document ID {doc_id}"
                        )
                    observed_docs[doc_id] = text_hash
                    prefix["documents"] += 1
                    yield doc

        packed = iter(pack_documents(documents(), CountingTokenizer(), seq_len))
        for index in range(count):
            try:
                inputs, labels, positions, document_ids = next(packed)
            except StopIteration as error:
                raise ValueError(
                    f"Validation source is too small: {index * seq_len} packed tokens, need {token_budget}"
                ) from error
            for key, values in zip(arrays, (inputs, labels, positions)):
                arrays[key][index] = values
            used_ids.update(dict.fromkeys(document_ids))
        try:
            next(packed)
        except StopIteration as error:
            raise ValueError(
                "Cannot prove validation source exceeds the budget by at least one full sequence"
            ) from error
        array_info = {}
        for key, array in arrays.items():
            array.flush()
            array._mmap.close()
            path = temporary / f"{key}.npy"
            array_info[key] = {"file": path.name, "sha256": file_sha256(path)}
        index_path = temporary / "evaluation_documents.jsonl"
        with index_path.open("w") as stream:
            for doc_id in used_ids:
                stream.write(
                    json.dumps({"id": doc_id, "text_sha256": observed_docs[doc_id]}, sort_keys=True)
                    + "\n"
                )
        metadata = {
            "schema_version": 1,
            "token_budget": token_budget,
            "sequence_length": seq_len,
            "num_sequences": count,
            "dtype": "int64",
            "arrays": array_info,
            "tokenizer_files": tokenizer_files,
            "source": {
                "repo_id": REPO_ID,
                "revision": REVISION,
                "config": CONFIG,
                **source,
                "rows": parquet.metadata.num_rows,
                "official_gpt2_tokens": official_tokens,
            },
            "source_manifest_sha256": file_sha256(manifest_path or MANIFEST_PATH),
            "packing": "TorchTitan document-local label shift; BOS/EOS; reset and chunk-normalize positions; source parquet row order; global packing before DP split",
            "packing_implementation_sha256": file_sha256(__file__),
            "available_packed_label_tokens_lower_bound": token_budget + seq_len,
            "raw_prefix_label_tokens": prefix["label_tokens"],
            "prefix_documents_tokenized": prefix["documents"],
            "evaluation_documents": {
                "file": index_path.name,
                "sha256": file_sha256(index_path),
                "count": len(used_ids),
                "text_normalization": NORMALIZATION,
            },
            "training_exclusion": "Entire source shard omitted from training; evaluated document IDs and normalized text hashes additionally filtered from every training shard",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing validation artifact: {output}")
        temporary.rename(output)
        return metadata
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
