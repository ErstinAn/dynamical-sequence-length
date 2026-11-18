"""Freeze document orders and validation without copying source text or tokens."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_packed_validation import prepare_packed
from scripts.prepare_document_validation import prepare_document_windows
from scripts.dataset_card import write_dataset_card


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def split_metadata(root, manifest, split):
    entries = [f for f in manifest["files"] if f["split"] == split]
    lengths, supplements, locations = [], [], {}
    for entry in entries:
        index = np.load(root / entry["index"]["path"], mmap_mode="r")
        values = pq.read_table(root / entry["text"]["path"], columns=["source_tokens"])
        pool = values["source_tokens"].to_numpy() < 16384
        if len(pool) != len(index):
            raise ValueError("Text rows and token index rows differ")
        lengths.extend(map(int, index[:, 1]))
        supplements.extend(map(bool, pool))
        locations[entry["text"]["path"]] = (len(lengths) - len(index), len(index))
    return np.asarray(lengths, dtype=np.int64), np.asarray(supplements), locations


def distribution(lengths, supplement, weights=None):
    weights = lengths if weights is None else weights
    total = int(weights.sum())
    bins = np.asarray([0, 16384, 24576, 32768, 65536, 131072, np.iinfo(np.int64).max])
    selected = lengths[weights > 0]
    return {
        "documents": int(np.count_nonzero(weights)), "raw_tokens": total,
        "supplement_raw_tokens": int(weights[supplement].sum()),
        "supplement_token_fraction": float(weights[supplement].sum() / total),
        "length_token_fractions": [float(weights[(lengths >= a) & (lengths < b)].sum() / total)
                                   for a, b in zip(bins[:-1], bins[1:])],
        "length_bin_edges": [0, 16384, 24576, 32768, 65536, 131072, None],
        "document_length": {
            "unit": "raw Llama 3 tokens; excludes BOS and EOS",
            "population": "distinct source documents with at least one selected raw input token",
            "minimum": int(selected.min()), "maximum": int(selected.max()),
            "mean": float(selected.mean()),
            "quantiles": {str(q): float(np.quantile(selected, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
            "bin_documents": [int(np.count_nonzero((selected >= a) & (selected < b)))
                              for a, b in zip(bins[:-1], bins[1:])],
        },
    }


def order_statistics(lengths, supplement, order):
    blocks = np.array_split(order, min(32, len(order)))
    return {
        "ordered_block_definition": "32 contiguous equal-document-count blocks, or fewer for a small dataset",
        "ordered_block_supplement_token_fractions": [
            float(lengths[b][supplement[b]].sum() / lengths[b].sum()) for b in blocks],
        "ordered_block_document_length_means": [float(lengths[b].mean()) for b in blocks],
        "dp4_ordered_block_document_length_means": [
            [float(lengths[b].mean()) for b in np.array_split(order[r::4], min(32, len(order[r::4])))]
            if len(order[r::4]) else [] for r in range(4)],
    }


def fixed_validation_statistics(validation_path, lengths, supplement, locations, full):
    val_manifest = json.loads((validation_path / "manifest.json").read_text())
    weights, occurrences = np.zeros_like(lengths), np.zeros_like(lengths)
    records = json.loads((validation_path / "documents.json").read_text())
    for record in records:
        seen = set()
        for fragment in record["fragments"]:
            doc = locations[fragment["file"]][0] + fragment["row"]
            start = fragment["document_token_offset"]
            end = start + fragment["length"]
            weights[doc] += max(0, min(end, int(lengths[doc]) + 1) - max(start, 1))
            seen.add(doc)
        for doc in seen:
            occurrences[doc] += 1
    selected = distribution(lengths, supplement, weights)
    selected.update({
        "original_documents": int(np.count_nonzero(weights[~supplement])),
        "supplement_documents": int(np.count_nonzero(weights[supplement])),
        "length_token_fraction_differences": [a - b for a, b in zip(
            selected["length_token_fractions"], full["length_token_fractions"])],
        "num_sequences": val_manifest["num_sequences"], "sequence_length": val_manifest["sequence_length"],
        "valid_labels": val_manifest["token_budget"], "padding_positions": val_manifest["padding_positions"],
        "manifest_sha256": digest(validation_path / "manifest.json"),
        "documents_with_multiple_selected_windows": int((occurrences > 1).sum()),
        "max_selected_windows_per_document": int(occurrences.max()),
    })
    single_document = val_manifest.get("packing") == "none"
    selected["representative_of_full_validation"] = not single_document
    selected["selection_population"] = val_manifest["sampling"]["kind"]
    if single_document:
        eligible = lengths >= val_manifest["sampling"]["minimum_raw_tokens"]
        selected["eligible_original_documents"] = int((eligible & ~supplement).sum())
        selected["eligible_supplement_documents"] = int((eligible & supplement).sum())
    elif (abs(selected["supplement_token_fraction"] - full["supplement_token_fraction"]) > 0.02
          or max(map(abs, selected["length_token_fraction_differences"])) > 0.03):
        raise ValueError("Fixed packed sample exceeds the source/length representation tolerances")
    return selected


def prepare(source, output, tokenizer, *, train_seed=42, validation_seed=43, sample_seed=44,
            length=16384, count=936, packing="none", offset_seed=45):
    source, output, tokenizer = map(lambda p: Path(p).resolve(), (source, output, tokenizer))
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((source / "manifest.json").read_text())
    original_verification = json.loads((source / "verification.json").read_text())
    source_hash = digest(source / "manifest.json")
    if not original["complete"] or not original_verification["passed"] or source_hash != original_verification["manifest_sha256"]:
        raise ValueError("Source dataset verification does not match its manifest")
    if original.get("document_orders"):
        raise ValueError("Supply the original dataset, not an already ordered view")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for name in ("data", "tokens"):
            (temporary / name).symlink_to(source / name, target_is_directory=True)
        (temporary / "orders").mkdir()
        orders, report, metadata = {}, {}, {}
        for split, seed in (("train", train_seed), ("validation", validation_seed)):
            lengths, supplement, locations = split_metadata(source, original, split)
            order = np.random.Generator(np.random.PCG64(seed)).permutation(len(lengths)).astype("<u4")
            path = temporary / "orders" / f"{split}.npy"
            np.save(path, order)
            orders[split] = {"path": str(path.relative_to(temporary)), "sha256": digest(path),
                             "documents": len(order), "seed": seed, "generator": "numpy.PCG64"}
            stats = distribution(lengths, supplement)
            stats["order_sha256"] = orders[split]["sha256"]
            stats["dp4_physical_positions"] = [int((lengths[order[r::4]] + 2).sum()) for r in range(4)]
            if split == "train":
                stats.update(order_statistics(lengths, supplement, order))
            report[split] = stats
            metadata[split] = (lengths, supplement, locations)
            print(json.dumps({"stage": "document_order", "split": split, **stats}), flush=True)
        manifest = {**original, "document_orders": orders,
                    "document_order_policy": "fixed-permutation-before-dp-split-v1",
                    "source_manifest_sha256": source_hash}
        write_json(temporary / "manifest.json", manifest)
        shared_files = []
        for entry in manifest["files"]:
            shared_files.append({**{k: v for k, v in entry.items() if k not in ("token_ids", "index")},
                                 "expected_token_ids_sha256": entry["token_ids"]["sha256"],
                                 "expected_index_sha256": entry["index"]["sha256"]})
        write_json(temporary / "dataset_manifest.json", {**manifest, "files": shared_files})
        verification = {"passed": True, "manifest_sha256": digest(temporary / "manifest.json"),
                        "dataset_manifest_sha256": digest(temporary / "dataset_manifest.json"),
                        "source_manifest_sha256": source_hash,
                        "source_verification_sha256": digest(source / "verification.json"),
                        "source_files_unchanged": True,
                        "split_membership_unchanged": True, "orders_are_complete_permutations": True,
                        "totals": original_verification["totals"],
                        "url_group_overlap": original_verification["url_group_overlap"]}
        write_json(temporary / "verification.json", verification)
        for name in ("selection.json", "tokenizer_provenance.json", "upload_receipt.json"):
            if (source / name).exists():
                shutil.copy2(source / name, temporary / ("source_upload_receipt.json" if name == "upload_receipt.json" else name))
        for name in ("rebuild_long_token_cache.py", "restore_final_dataset.py"):
            shutil.copy2(Path(__file__).with_name(name), temporary / name)
        validation_path = temporary / "validation"
        if packing == "none":
            val_manifest = prepare_document_windows(temporary, validation_path, tokenizer, length, count,
                                                    sample_seed=sample_seed, offset_seed=offset_seed)
        elif packing == "stream":
            val_manifest = prepare_packed(temporary, validation_path, tokenizer, length, count, sample_seed=sample_seed)
        else:
            raise ValueError("Validation packing must be none or stream")
        lengths, supplement, locations = metadata["validation"]
        report["fixed_validation"] = fixed_validation_statistics(
            validation_path, lengths, supplement, locations, report["validation"])
        report["dataset_manifest_sha256"] = verification["manifest_sha256"]
        write_json(temporary / "statistics.json", report)
        bundle = "${ASSETS_DIR}/data/" + output.name
        write_json(temporary / "training_settings.json", {
            "dataloader.dataset_path": bundle,
            "dataloader.expected_manifest_sha256": verification["manifest_sha256"],
            "dataloader.packing": "stream", "dataloader.infinite": False,
            "validator.dataloader.dataset_path": bundle,
            "validator.dataloader.expected_manifest_sha256": verification["manifest_sha256"],
            "validator.dataloader.packing": packing,
            "validator.artifact_path": bundle + "/validation",
            "validator.eval_seq_len": length, "validator.token_budget": val_manifest["token_budget"],
            "validator.eval_local_batch_size": 2,
        })
        write_dataset_card(temporary)
        temporary.rename(output)
        print(json.dumps({"stage": "complete", "output": str(output), **report["fixed_validation"]}), flush=True)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--validation-seed", type=int, default=43)
    p.add_argument("--sample-seed", type=int, default=44)
    p.add_argument("--offset-seed", type=int, default=45)
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--count", type=int, default=936)
    p.add_argument("--validation-packing", choices=("none", "stream"), default="none")
    a = p.parse_args()
    prepare(a.source, a.output, a.tokenizer, train_seed=a.train_seed,
            validation_seed=a.validation_seed, sample_seed=a.sample_seed, offset_seed=a.offset_seed,
            length=a.length, count=a.count, packing=a.validation_packing)
