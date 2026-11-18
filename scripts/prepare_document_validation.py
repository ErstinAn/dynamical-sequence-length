"""Freeze nonoverlapping validation windows within individual documents."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_packed_validation import digest
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataset

PROTOCOL = "single-document-disjoint-blocks-random-offset-v1"


def prepare_document_windows(dataset, output, tokenizer, length=16384, count=936,
                             *, sample_seed=44, offset_seed=45):
    dataset, output, tokenizer = map(Path, (dataset, output, tokenizer))
    if output.exists():
        raise FileExistsError(output)
    if length <= 0 or count <= 0:
        raise ValueError("Validation length and count must be positive")
    source = LongDocumentDataset(dataset, split="validation", seq_len=length)
    if digest(tokenizer / "tokenizer.json") != source.tokenizer_sha256:
        raise ValueError("Tokenizer differs from source corpus")
    lengths = np.concatenate([index[:, 1] for index in source.indexes]).astype(np.int64)
    order = source.document_order
    if order is None:
        order = np.arange(len(lengths))
    # Each block contains L input tokens and one next-token target. Blocks do
    # not share even the final lookahead token. BOS/EOS are virtual doc tokens.
    width = length + 1
    counts = (lengths + 2) // width
    offsets = np.random.Generator(np.random.PCG64(offset_seed))
    candidates = []
    for doc in map(int, order):
        windows = int(counts[doc])
        if not windows:
            continue
        slack = int(lengths[doc]) + 2 - windows * width
        start = int(offsets.integers(slack + 1))
        candidates.extend((doc, start + window * width) for window in range(windows))
    if count > len(candidates):
        raise ValueError(f"Only {len(candidates)} disjoint single-document windows are available; requested {count}")
    selected = np.sort(np.random.Generator(np.random.PCG64(sample_seed)).choice(
        len(candidates), size=count, replace=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        arrays = {name: np.lib.format.open_memmap(temporary / f"{name}.npy", mode="w+",
                  dtype=np.int64, shape=(count, length)) for name in ("input", "labels", "positions")}
        positions = np.arange(length, dtype=np.int64)
        documents, maps = [], {}
        for row, candidate_id in enumerate(map(int, selected)):
            doc, start = candidates[candidate_id]
            file_idx = int(np.searchsorted(source.starts, doc, side="right") - 1)
            source_row = int(doc - source.starts[file_idx])
            entry = source.files[file_idx]
            if file_idx not in maps:
                maps[file_idx] = np.memmap(dataset / entry["token_ids"]["path"], mode="r", dtype="<u4")
            token_start, size = map(int, source.indexes[file_idx][source_row])
            doc_positions = np.arange(start, start + width, dtype=np.int64)
            values = np.empty(width, dtype=np.int64)
            raw = (doc_positions >= 1) & (doc_positions <= size)
            values[raw] = maps[file_idx][token_start + doc_positions[raw] - 1]
            values[doc_positions == 0] = source.bos
            values[doc_positions == size + 1] = source.eos
            arrays["input"][row], arrays["labels"][row] = values[:-1], values[1:]
            arrays["positions"][row] = positions
            documents.append({
                "sequence_id": candidate_id, "document_id": doc,
                "fragments": [{"file": entry["text"]["path"], "row": source_row,
                               "document_token_offset": start, "length": length}],
                "lookahead": {"file": entry["text"]["path"], "row": source_row,
                              "document_token_offset": start + length},
            })
        for array in arrays.values():
            array.flush()
        (temporary / "documents.json").write_text(json.dumps(documents, indent=2) + "\n")
        manifest = {
            "schema_version": 1, "sequence_length": length, "num_sequences": count,
            "token_budget": count * length, "physical_positions": count * length,
            "vocab_size": source.vocab_size, "packing_protocol": PROTOCOL,
            "packing": "none", "packed_sequences": 0, "ignored_boundary_labels": 0,
            "padding_positions": 0,
            "sampling": {
                "kind": "uniform_disjoint_document_windows_without_replacement",
                "seed": sample_seed, "offset_seed": offset_seed, "generator": "numpy.PCG64",
                "candidate_sequences": len(candidates), "eligible_documents": int((counts > 0).sum()),
                "excluded_short_documents": int((counts == 0).sum()),
                "minimum_raw_tokens": length - 1,
                "block_width_including_lookahead": width,
                "selected_sequence_ids": selected.tolist(),
                "multiple_windows_per_document": True,
                "overlap_including_lookahead": False,
                "offset_rule": "Uniform shared starting offset within each document's unused slack, inclusive",
            },
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
    print(json.dumps({"output": str(output), "sequences": count, "valid_labels": count * length,
                      "eligible_documents": int((counts > 0).sum()), "candidate_windows": len(candidates)}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--count", type=int, default=936)
    parser.add_argument("--sample-seed", type=int, default=44)
    parser.add_argument("--offset-seed", type=int, default=45)
    args = parser.parse_args()
    prepare_document_windows(args.dataset, args.output, args.tokenizer, args.length, args.count,
                             sample_seed=args.sample_seed, offset_seed=args.offset_seed)
