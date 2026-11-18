"""Keep historical assets resolvable after moving them out of assets/data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torchtitan_dynamic_seq_len.data.asset_paths import resolve_archived_asset


class ArchivedDataPathTests(unittest.TestCase):
    def test_exact_local_alias_only_and_existing_directory_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "assets/data/old_validation"
            archived = root / "archive/data/20261116/old_validation"
            archived.mkdir(parents=True)
            (root / "archive/data/manifest.json").write_text(json.dumps({
                "entries": {"old_validation": {"directory": "20261116/old_validation"}}}))
            with patch("torchtitan_dynamic_seq_len.data.asset_paths.ROOT", root):
                self.assertEqual(resolve_archived_asset(old), archived)
                unrelated = root / "elsewhere/old_validation"
                self.assertEqual(resolve_archived_asset(unrelated), unrelated)
                old.mkdir(parents=True)
                self.assertEqual(resolve_archived_asset(old), old)

    def test_missing_archive_does_not_substitute_new_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "assets/data/old_validation"
            with patch("torchtitan_dynamic_seq_len.data.asset_paths.ROOT", root):
                self.assertEqual(resolve_archived_asset(old), old)


if __name__ == "__main__":
    unittest.main()
