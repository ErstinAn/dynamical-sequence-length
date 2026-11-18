"""Freeze packed validation windows and their exact valid label budget."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan_dynamic_seq_len.data.packed_documents import PACKING_PROTOCOL, PackedDocumentDataset


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_packed(dataset, output, tokenizer, length=22256, count=936, *, sample_seed=None):
    dataset, output, tokenizer = map(Path, (dataset, output, tokenizer))
    if output.exists():
        raise FileExistsError(output)
    source = PackedDocumentDataset(dataset, split="validation", seq_len=length)
    if digest(tokenizer / "tokenizer.json") != source.tokenizer_sha256:
        raise ValueError("Tokenizer differs from source corpus")
    if count <= 0:
        raise ValueError("Specify a positive fixed validation sequence count")
    sampling = {"kind": "stream_prefix"}
    selected = set(range(count))
    if sample_seed is not None:
        physical = sum(int(index[:, 1].sum()) + 2 * len(index) for index in source.indexes)
        candidates = (physical + length - 1) // length
        if physical % length == 1:  # An EOS-only final window has no prediction targets.
            candidates -= 1
        if count > candidates:
            raise ValueError("Requested sample count exceeds the complete validation sequence pool")
        selected = set(map(int, np.random.Generator(np.random.PCG64(sample_seed)).choice(
            candidates, size=count, replace=False)))
        sampling = {"kind": "uniform_sequences_without_replacement", "seed": sample_seed,
                    "generator": "numpy.PCG64", "candidate_sequences": candidates,
                    "selected_sequence_ids": sorted(selected)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        arrays = {name: np.lib.format.open_memmap(temporary / f"{name}.npy", mode="w+",
                  dtype=np.int64, shape=(count, length)) for name in ("input", "labels", "positions")}
        fragments, valid, boundary, padding = [], 0, 0, 0
        iterator = iter(source)
        row = 0
        for sequence_id in range(max(selected) + 1):
            previous_boundary, previous_padding = source.boundary_labels, source.padding_positions
            try:
                inputs, labels = next(iterator)
            except StopIteration as exc:
                raise ValueError("Validation stream cannot supply the requested sequence count") from exc
            if sequence_id not in selected:
                continue
            arrays["input"][row] = inputs["input"].numpy()
            arrays["positions"][row] = inputs["positions"].numpy()
            arrays["labels"][row] = labels.numpy()
            valid += int((labels != IGNORE_INDEX).sum())
            boundary += source.boundary_labels - previous_boundary
            padding += source.padding_positions - previous_padding
            fragments.append({"sequence_id": sequence_id, "fragments": source.last_fragments,
                              "lookahead": source.last_lookahead})
            row += 1
        for array in arrays.values():
            array.flush()
        (temporary / "documents.json").write_text(json.dumps(fragments, indent=2) + "\n")
        manifest = {
            "schema_version": 2, "sequence_length": length, "num_sequences": count,
            "token_budget": valid, "physical_positions": count * length,
            "vocab_size": source.vocab_size, "packing_protocol": PACKING_PROTOCOL,
            "ignored_boundary_labels": boundary,
            "padding_positions": padding, "sampling": sampling,
            "source": {"repo_id": "long-sequence-length-dataset/fineweb-edu-long-30b",
                       "split": "validation", "manifest_sha256": source.manifest_hash},
            "tokenizer_files": {name: digest(tokenizer / name)
                                for name in ("tokenizer.json", "tokenizer_config.json")},
            "arrays": {name: {"file": f"{name}.npy", "sha256": digest(temporary / f"{name}.npy")}
                       for name in arrays},
            "evaluation_documents": {"file": "documents.json", "sha256": digest(temporary / "documents.json")},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(json.dumps({"output": str(output), "sequences": count, "valid_labels": valid}))
    return manifest
