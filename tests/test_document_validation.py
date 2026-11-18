"""Check fixed windows that never share source tokens or cross documents."""

import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.prepare_document_validation import prepare_document_windows
from tests.test_document_packing import fixture
from torchtitan_dynamic_seq_len.data.fixed_validation import FrozenValidationArtifact


class DocumentValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docs, self.tokenizer = fixture(self.root)

    def test_multiple_windows_per_document_without_overlap_including_targets(self):
        path = self.root / "fixed"
        manifest = prepare_document_windows(self.root, path, self.tokenizer, length=8, count=6)
        frozen = FrozenValidationArtifact(path, token_budget=48, sequence_length=8, tokenizer_path=self.tokenizer)
        records = json.loads((path / "documents.json").read_text())
        intervals = defaultdict(list)
        for i, record in enumerate(records):
            self.assertEqual(len(record["fragments"]), 1)
            fragment = record["fragments"][0]
            doc = record["document_id"]
            start = fragment["document_token_offset"]
            whole = np.asarray([254] + self.docs[doc] + [255])
            expected = whole[start:start + 9]
            np.testing.assert_array_equal(frozen.arrays["input"][i], expected[:-1])
            np.testing.assert_array_equal(frozen.arrays["labels"][i], expected[1:])
            np.testing.assert_array_equal(frozen.arrays["positions"][i], np.arange(8))
            self.assertEqual(record["lookahead"]["row"], fragment["row"])
            self.assertEqual(record["lookahead"]["file"], fragment["file"])
            intervals[doc].append((start, start + 9))
        self.assertEqual(max(map(len, intervals.values())), 3)
        for spans in intervals.values():
            spans.sort()
            for left, right in zip(spans, spans[1:]):
                self.assertLessEqual(left[1], right[0])
        self.assertEqual(manifest["sampling"]["candidate_sequences"], 6)
        self.assertEqual(manifest["sampling"]["excluded_short_documents"], 2)
        self.assertEqual(manifest["packing"], "none")
        self.assertEqual(manifest["padding_positions"], 0)

    def test_seed_reproduces_exact_arrays_and_random_offsets(self):
        a = prepare_document_windows(self.root, self.root / "a", self.tokenizer, length=8, count=5)
        b = prepare_document_windows(self.root, self.root / "b", self.tokenizer, length=8, count=5)
        self.assertEqual(a, b)
        records = json.loads((self.root / "a/documents.json").read_text())
        self.assertTrue(any(r["fragments"][0]["document_token_offset"] > 0 for r in records))

    def test_insufficient_pool_fails_without_partial_dataset(self):
        path = self.root / "too_many"
        with self.assertRaisesRegex(ValueError, "Only 6"):
            prepare_document_windows(self.root, path, self.tokenizer, length=8, count=7)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
