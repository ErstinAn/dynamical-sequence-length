"""Check restoration after the old Hub revision has been removed."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.restore_final_dataset import restore


class RestoreFinalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shared = {"files": [{"text": {"path": "data/train-00000.parquet"}}]}
        self.write("dataset_manifest.json", self.shared)
        self.manifest = b'{"complete": true}\n'
        self.write("verification.json", {
            "passed": True,
            "dataset_manifest_sha256": self.digest(self.root / "dataset_manifest.json"),
            "manifest_sha256": hashlib.sha256(self.manifest).hexdigest(),
        })
        self.write("source_upload_receipt.json", {"repo_id": "test/data", "revision": "removed-old-commit"})
        self.write("tokenizer_provenance.json", {"repo_id": "test/tokenizer", "revision": "tokenizer-commit",
                                               "files_sha256": {"tokenizer.json": "unused"}})

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_current_revision_is_resolved_once_and_pinned_for_all_data_reads(self):
        fake_rebuild = SimpleNamespace(rebuild=lambda root, tokenizer: (root / "manifest.json").write_bytes(self.manifest))
        with patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.hf_hub_download", return_value=str(self.root / "dataset_manifest.json")) as metadata, \
             patch("huggingface_hub.snapshot_download") as download, \
             patch.dict("sys.modules", {"scripts.rebuild_long_token_cache": fake_rebuild}):
            api.return_value.dataset_info.return_value.sha = "current-commit"
            restore(self.root, self.root / "tokenizer")
            api.return_value.dataset_info.assert_called_once_with("test/data", revision=None)
            self.assertEqual(metadata.call_args.kwargs["revision"], "current-commit")
            self.assertEqual(download.call_args_list[0].kwargs["revision"], "current-commit")
            self.assertEqual(download.call_args_list[1].kwargs["revision"], "tokenizer-commit")
            self.assertEqual(download.call_args_list[0].kwargs["allow_patterns"], ["data/train-00000.parquet"])

    def test_mismatched_remote_package_stops_before_large_download_or_rebuild(self):
        remote = self.root / "different.json"
        remote.write_text("{}")
        with patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.hf_hub_download", return_value=str(remote)), \
             patch("huggingface_hub.snapshot_download") as download:
            api.return_value.dataset_info.return_value.sha = "newer-commit"
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                restore(self.root, self.root / "tokenizer", revision="requested-commit")
            api.return_value.dataset_info.assert_called_once_with("test/data", revision="requested-commit")
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
