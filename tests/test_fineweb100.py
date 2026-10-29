"""Protocol tests for the complete training split and frozen validation artifact."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from torchtitan_dynamic_seq_len.data import fineweb100 as fw
from torchtitan_dynamic_seq_len.data import fineweb100_identity as identity


class IntegerTokenizer:
    def encode(self, text, add_bos=True, add_eos=True):
        return (
            ([90] if add_bos else [])
            + [int(word) for word in text.split()]
            + ([99] if add_eos else [])
        )


class FineWeb100Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}")
        self.documents = [
            {"id": str(index), "text": text, "token_count": len(text.split())}
            for index, text in enumerate(
                ("1 2", "3 4 5 6 7 8 9 10 11 12 13", "14", "15 16 17 18", "19 20 21")
            )
        ]
        self.source = self.root / "source.parquet"
        pq.write_table(pa.Table.from_pylist(self.documents), self.source)
        self.manifest = {
            "schema_version": 1,
            "repo_id": fw.REPO_ID,
            "revision": fw.REVISION,
            "config": fw.CONFIG,
            "source_files": [
                {
                    "path": f"sample/100BT/{group:03d}_{part:05d}.parquet",
                    "size": 1,
                    "sha256": "0" * 64,
                }
                for group in range(15)
                for part in range(10)
            ],
            "validation_files": [fw.VALIDATION_FILE],
            "training_files": [
                f"sample/100BT/{group:03d}_{part:05d}.parquet"
                for group in range(15)
                for part in range(10)
                if (group, part) != (0, 0)
            ],
        }
        self.manifest["source_files"][0].update(
            size=self.source.stat().st_size, sha256=fw.file_sha256(self.source)
        )
        self.manifest_path = self.root / "split.json"
        self.write_manifest()
        self.manifest_patch = patch.object(fw, "MANIFEST_PATH", self.manifest_path)
        self.manifest_patch.start()
        self.addCleanup(self.manifest_patch.stop)
        self.reference_path = self.root / "reference.json"
        reference_patch = patch.object(identity, "REFERENCE_PATH", self.reference_path)
        reference_patch.start()
        self.addCleanup(reference_patch.stop)

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def prepare(self, output=None, budget=16):
        metadata = fw.prepare_validation(
            self.source,
            output or self.root / "artifact",
            self.tokenizer,
            budget,
            4,
            tokenizer=IntegerTokenizer(),
        )
        self.reference_path.write_text(
            json.dumps({"artifact_identity": identity.validation_identity(metadata)})
        )
        return metadata

    def test_exact_full_split_is_pinned_and_disjoint(self):
        manifest = fw.load_manifest()
        files = fw.training_data_files(fw.REPO_ID)
        self.assertEqual(len(files), 149)
        self.assertTrue(
            all(
                path.startswith(f"hf://datasets/{fw.REPO_ID}@{fw.REVISION}/sample/100BT/")
                for path in files
            )
        )
        self.assertTrue(files[0].endswith("000_00001.parquet"))
        self.assertTrue(files[-1].endswith("014_00009.parquet"))
        self.assertFalse(set(manifest["training_files"]) & set(manifest["validation_files"]))
        for change in (
            lambda: self.manifest["training_files"].pop(),
            lambda: self.manifest["training_files"].append(fw.VALIDATION_FILE),
        ):
            original = list(self.manifest["training_files"])
            change()
            self.write_manifest()
            with self.assertRaises(ValueError):
                fw.load_manifest()
            self.manifest["training_files"] = original

    def test_incomplete_local_snapshot_does_not_become_prefix_training(self):
        with self.assertRaisesRegex(FileNotFoundError, "149 training shards missing"):
            fw.training_data_files(str(self.root))

    def test_packing_matches_actual_upstream_dataset(self):
        from datasets import Dataset
        from torchtitan.hf_datasets import DatasetConfig
        from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataset

        name = "fineweb100_packing_test"
        with patch.dict(
            DATASETS,
            {
                name: DatasetConfig(
                    path="memory",
                    loader=lambda _: Dataset.from_list(self.documents).to_iterable_dataset(),
                    sample_processor=lambda doc: doc["text"],
                )
            },
        ):
            expected = list(HuggingFaceTextDataset(name, "memory", IntegerTokenizer(), seq_len=4))
        actual = list(fw.pack_documents(self.documents, IntegerTokenizer(), 4))
        self.assertEqual(len(actual), len(expected))
        for (inputs, labels), (x, y, positions, doc_ids) in zip(expected, actual):
            self.assertEqual(inputs["input"].tolist(), x)
            self.assertEqual(labels.tolist(), y)
            self.assertEqual(inputs["positions"].tolist(), positions)
            self.assertTrue(doc_ids)

    def test_artifact_budget_source_capacity_and_hashes(self):
        metadata = self.prepare()
        output = self.root / "artifact"
        self.assertEqual(metadata["token_budget"], 16)
        self.assertEqual(metadata["num_sequences"], 4)
        self.assertGreater(metadata["available_packed_label_tokens_lower_bound"], 16)
        self.assertGreaterEqual(metadata["raw_prefix_label_tokens"], 20)
        self.assertEqual(
            metadata["source"]["official_gpt2_tokens"],
            sum(doc["token_count"] for doc in self.documents),
        )
        expected = list(fw.pack_documents(self.documents, IntegerTokenizer(), 4))[:4]
        for column, key in enumerate(("input", "labels", "positions")):
            entry = metadata["arrays"][key]
            self.assertEqual(fw.file_sha256(output / entry["file"]), entry["sha256"])
            actual = np.load(output / entry["file"])
            self.assertEqual(actual.dtype, np.dtype("int64"))
            self.assertEqual(actual.tolist(), [sample[column] for sample in expected])
        used_ids, hashes = fw.load_evaluation_exclusions(output)
        self.assertEqual(used_ids, {doc_id for sample in expected for doc_id in sample[3]})
        self.assertEqual(
            hashes,
            {
                fw.normalized_text_hash(doc["text"])
                for doc in self.documents
                if doc["id"] in used_ids
            },
        )
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_loader_never_opens_validation_and_filters_copies(self):
        self.prepare()
        source_docs = [
            self.documents[0],
            {"id": "different-id", "text": " 1\t2\n"},
            {"id": "keep-me", "text": "42 43"},
        ]

        class FakeDataset:
            def filter(self, predicate):
                return [doc for doc in source_docs if predicate(doc)]

        with (
            patch.dict(
                "os.environ", {"DYNAMIC_SEQ_LEN_VALIDATION_DIR": str(self.root / "artifact")}
            ),
            patch("datasets.load_dataset", return_value=FakeDataset()) as loader,
        ):
            result = fw.load_fineweb100(fw.REPO_ID)
        self.assertEqual(result, [source_docs[2]])
        paths = loader.call_args.kwargs["data_files"]["train"]
        self.assertEqual(len(paths), 149)
        self.assertFalse(any(path.endswith(fw.VALIDATION_FILE) for path in paths))
        self.assertTrue(loader.call_args.kwargs["streaming"])

    def test_source_hash_failure_leaves_no_output(self):
        self.manifest["source_files"][0]["sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare()
        self.assertFalse((self.root / "artifact").exists())

    def test_insufficient_tokens_fails_atomically(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            self.prepare(budget=256)
        self.assertFalse((self.root / "artifact").exists())
        self.assertFalse(list(self.root.glob(".artifact.*")))

    def test_missing_or_tampered_exclusion_index_fails_closed(self):
        with self.assertRaisesRegex(FileNotFoundError, "Prepare the frozen"):
            fw.load_evaluation_exclusions(self.root / "missing")
        self.prepare()
        (self.root / "artifact/evaluation_documents.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "checksum"):
            fw.load_evaluation_exclusions(self.root / "artifact")

    def test_different_artifact_identity_is_rejected_before_training(self):
        metadata = self.prepare()
        metadata["arrays"]["input"]["sha256"] = "0" * 64
        (self.root / "artifact/manifest.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "differs from the versioned reference: arrays"):
            fw.load_evaluation_exclusions(self.root / "artifact")


if __name__ == "__main__":
    unittest.main()
