"""Check fresh indexes and reconstruct every fixed validation window.

Run this read-only check after the full cache reconstruction succeeds. It uses
the selected cache and fixed metadata. It does not require a Slurm readiness file.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download_dataset import check_metadata, require_hash, sha256, standalone_paths, TRAINING_MANIFEST_SHA256


def verify(dataset, tokenizer):
    if sys.flags.optimize:
        raise RuntimeError("Run this check without Python optimization so all assertions execute.")
    root, tokenizer = standalone_paths(dataset, tokenizer)
    progress = json.loads((root / "rebuild_progress.json").read_text())
    assert progress["complete"] is True
    assert progress["manifest_sha256"] == TRAINING_MANIFEST_SHA256
    ready_path = root / "standalone_ready.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        assert ready["complete"] is True
        assert ready["manifest_sha256"] == TRAINING_MANIFEST_SHA256
    shared, _ = check_metadata(root, tokenizer)
    manifest = json.loads((root / "manifest.json").read_text())
    validation_files, counts = {}, Counter()
    validation_start = 0
    for entry in manifest["files"]:
        index_path = root / entry["index"]["path"]
        require_hash(index_path, entry["index"]["sha256"])
        index = np.load(index_path, mmap_mode="r")
        assert index.shape == (entry["documents"], 2)
        expected_offsets = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(index[:-1, 1], dtype=np.uint64)))
        assert np.array_equal(index[:, 0], expected_offsets)
        assert int(index[:, 1].sum()) == entry["tokens"]
        assert (root / entry["token_ids"]["path"]).stat().st_size == entry["tokens"] * 4
        counts[entry["split"]] += entry["documents"]
        if entry["split"] == "validation":
            binary = root / entry["token_ids"]["path"]
            require_hash(binary, entry["token_ids"]["sha256"])
            validation_files[entry["text"]["path"]] = (
                np.memmap(binary, mode="r", dtype="<u4"), index, validation_start)
            validation_start += entry["documents"]
    for split, entry in shared["document_orders"].items():
        order = np.load(root / entry["path"], mmap_mode="r")
        assert len(order) == counts[split]
        assert np.array_equal(np.sort(order), np.arange(counts[split]))

    validation = json.loads((root / "validation/manifest.json").read_text())
    arrays = {name: np.load(root / "validation" / entry["file"], mmap_mode="r")
              for name, entry in validation["arrays"].items()}
    windows = json.loads((root / "validation/documents.json").read_text())
    assert len(windows) == 936
    assert all(array.shape == (936, 16384) for array in arrays.values())
    assert (validation["packing"], validation["padding_positions"]) == ("none", 0)
    position = np.arange(16384, dtype=np.int64)
    spans = defaultdict(list)
    bos, eos = manifest["tokenizer"]["bos_id"], manifest["tokenizer"]["eos_id"]
    for row, window in enumerate(windows):
        assert len(window["fragments"]) == 1
        fragment = window["fragments"][0]
        lookahead = window["lookahead"]
        assert fragment["length"] == 16384
        assert (fragment["file"], fragment["row"]) == (lookahead["file"], lookahead["row"])
        start = fragment["document_token_offset"]
        assert lookahead["document_token_offset"] == start + 16384
        tokens, index, first_document = validation_files[fragment["file"]]
        assert first_document + fragment["row"] == window["document_id"]
        offset, length = map(int, index[fragment["row"]])
        # Construct the complete virtual document, then slice one window.
        document = np.concatenate(([bos], tokens[offset:offset + length], [eos]))
        block = document[start:start + 16385]
        assert len(block) == 16385
        assert np.array_equal(arrays["input"][row], block[:-1])
        assert np.array_equal(arrays["labels"][row], block[1:])
        assert np.array_equal(arrays["positions"][row], position)
        assert np.all((arrays["labels"][row] >= 0) & (arrays["labels"][row] < 128256))
        spans[window["document_id"]].append((start, start + 16385))
    for ranges in spans.values():
        ordered = sorted(ranges)
        assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "dataset_dir": str(root),
            "python": sys.executable, "python_version": sys.version, "numpy_version": np.__version__,
            "checker_sha256": sha256(Path(__file__)), "python_optimization": sys.flags.optimize,
            "passed": True, "checked_indexes": len(manifest["files"]),
            "split_documents": dict(counts), "reconstructed_validation_windows": len(windows),
            "window_length": 16384, "valid_targets": int((arrays["labels"] != -100).sum()),
            "unique_validation_documents": len(spans),
            "documents_with_multiple_windows": sum(len(ranges) > 1 for ranges in spans.values()),
            "overlap_including_lookahead": False, "shifted_labels_match_new_cache": True,
            "positions_match": True, "all_index_offsets_contiguous": True,
            "order_permutations_complete": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset_dir, args.tokenizer_dir), indent=2))
