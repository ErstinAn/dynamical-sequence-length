"""Check standalone download safeguards and fixed metadata verification."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_dataset as download


class DownloadDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.tokenizer = self.root / "tokenizer"
        self.dataset.mkdir()
        self.tokenizer.mkdir()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return download.sha256(path)

    def fixture(self):
        order = self.dataset / "orders/train.npy"
        order.parent.mkdir()
        order.write_bytes(b"ordered document ids")
        shared = {"files": [], "document_orders": {
            "train": {"path": "orders/train.npy", "sha256": download.sha256(order)}}}
        shared_sha = self.write(self.dataset / "dataset_manifest.json", shared)
        manifest_sha = self.write(self.dataset / "manifest.json", {"complete": True})
        validation = {"num_sequences": 936, "sequence_length": 16384,
                      "token_budget": 15335424, "arrays": {}}
        for name in ("input", "labels", "positions", "documents"):
            path = self.dataset / "validation" / f"{name}.test"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
            entry = {"file": path.name, "sha256": download.sha256(path)}
            if name == "documents":
                validation["evaluation_documents"] = entry
            else:
                validation["arrays"][name] = entry
        validation_sha = self.write(self.dataset / "validation/manifest.json", validation)
        self.write(self.dataset / "verification.json", {
            "passed": True, "dataset_manifest_sha256": shared_sha,
            "manifest_sha256": manifest_sha})
        token_sha = self.write(self.tokenizer / "tokenizer.json", {"test": True})
        self.write(self.dataset / "tokenizer_provenance.json", {
            "revision": "pinned", "files_sha256": {"tokenizer.json": token_sha}})
        return {"DATASET_MANIFEST_SHA256": shared_sha,
                "TRAINING_MANIFEST_SHA256": manifest_sha,
                "VALIDATION_MANIFEST_SHA256": validation_sha}

    def test_hash_checks_cover_order_validation_and_tokenizer(self):
        identities = self.fixture()
        with patch.multiple(download, **identities):
            shared, provenance = download.check_metadata(self.dataset, self.tokenizer)
            self.assertIn("train", shared["document_orders"])
            self.assertEqual(provenance["revision"], "pinned")
            for path in (self.dataset / "orders/train.npy",
                         self.dataset / "validation/labels.test",
                         self.tokenizer / "tokenizer.json"):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    download.check_metadata(self.dataset, self.tokenizer)
                path.write_bytes(original)

    def test_refuses_original_data_and_token_links(self):
        for name in ("data", "tokens"):
            link = self.dataset / name
            link.symlink_to(self.root, target_is_directory=True)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "standalone"):
                download.standalone_paths(self.dataset, self.tokenizer)
            link.unlink()
        self.assertEqual(download.standalone_paths(self.dataset, self.tokenizer),
                         (self.dataset, self.tokenizer))


if __name__ == "__main__":
    unittest.main()
