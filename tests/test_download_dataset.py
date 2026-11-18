"""Check standalone download safeguards and fixed metadata verification."""

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
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
        token_sha = self.write(self.tokenizer / "tokenizer.json", {"test": True})
        source = {"revision": "pinned", "files_sha256": {"tokenizer.json": token_sha}}
        self.write(self.dataset / "tokenizer_provenance.json", source)
        order = self.dataset / "orders/train.npy"
        order.parent.mkdir()
        order.write_bytes(b"ordered document ids")
        shared = {"tokenizer": source, "files": [], "document_orders": {
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

    def test_bundled_tokenizer_installs_offline_and_rejects_modified_files(self):
        reference = json.loads((download.BUNDLED_TOKENIZER / "tokenizer_source.json").read_text())
        download.install_bundled_tokenizer(self.tokenizer, reference)
        for name, expected in reference["files_sha256"].items():
            self.assertEqual(download.sha256(self.tokenizer / name), expected)
        for name in ("LICENSE", "Notice", "USE_POLICY.md"):
            self.assertEqual((self.tokenizer / name).read_bytes(),
                             (download.BUNDLED_TOKENIZER / name).read_bytes())
        download.install_bundled_tokenizer(self.tokenizer, reference)
        (self.tokenizer / "tokenizer.json").write_text("modified")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            download.install_bundled_tokenizer(self.tokenizer, reference)
        with self.assertRaisesRegex(ValueError, "provenance"):
            download.install_bundled_tokenizer(self.tokenizer, {**reference, "revision": "other"})

    def test_hub_calls_use_anonymous_access_unless_requested(self):
        identities = self.fixture()
        repo = "Erstin/fineweb-edu-long-30b"
        with (
            patch.multiple(download, **identities),
            patch.dict(os.environ, {"HF_TOKEN": "unused-test-value"}),
            patch("huggingface_hub.HfApi.dataset_info") as info,
            patch("huggingface_hub.hf_hub_download", return_value=self.dataset / "dataset_manifest.json") as manifest,
            patch("huggingface_hub.snapshot_download") as snapshots,
            patch.object(download, "install_bundled_tokenizer"),
            redirect_stdout(StringIO()),
        ):
            info.return_value.sha = "pinned"
            for options in ({}, {"authenticated": True}):
                with self.subTest(options=options):
                    token = options.get("authenticated", False)
                    download.download(self.dataset, self.tokenizer, revision="pinned",
                                      cache_dir=self.root / "cache", **options)
                    info.assert_called_with(repo, revision="pinned", token=token)
                    self.assertEqual(snapshots.call_count, 2)
                    for call in [manifest.call_args, *snapshots.call_args_list]:
                        self.assertEqual(call.kwargs["repo_id"], repo)
                        self.assertEqual(call.kwargs["revision"], "pinned")
                        self.assertIs(call.kwargs["token"], token)
                    snapshots.reset_mock()

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
