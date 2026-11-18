"""Check byte identity, document order, and interrupted cache reconstruction."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts import rebuild_long_token_cache as cache


class RebuildCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.tokenizer_dir = self.base / "tokenizer"
        self.tokenizer_dir.mkdir()
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "red": 1, "green": 2, "blue": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.tokenizer_dir / "tokenizer.json"))
        (self.source / "data").mkdir()
        (self.source / "orders").mkdir()
        np.save(self.source / "orders/train.npy", np.arange(79, -1, -1, dtype=np.uint64))
        self.expected = {}
        entries = []
        expected_files = []
        for shard in range(2):
            name = f"train-{shard:05d}"
            texts = ["red green", "blue", "green red blue", ""] * 10
            ids_by_doc = [[1, 2], [3], [2, 1, 3], []] * 10
            rows = [{"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "llama3_tokens": len(ids)} for text, ids in zip(texts, ids_by_doc)]
            text_path = self.source / "data" / f"{name}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), text_path)
            binary_bytes = np.asarray([item for ids in ids_by_doc for item in ids], dtype="<u4").tobytes()
            offsets, count = [], 0
            for ids in ids_by_doc:
                offsets.append((count, len(ids)))
                count += len(ids)
            index_oracle = self.base / f"{name}.oracle.npy"
            np.save(index_oracle, np.asarray(offsets, dtype=np.uint64))
            index_bytes = index_oracle.read_bytes()
            self.expected[f"{name}.bin"] = binary_bytes
            self.expected[f"{name}.index.npy"] = index_bytes
            entry = {"split": "train", "documents": 40, "tokens": count,
                     "text": {"path": f"data/{name}.parquet", "sha256": cache.sha256(text_path)},
                     "expected_token_ids_sha256": hashlib.sha256(binary_bytes).hexdigest(),
                     "expected_index_sha256": hashlib.sha256(index_bytes).hexdigest()}
            entries.append(entry)
            expected_files.append({"split": "train", "documents": 40, "tokens": count,
                                   "text": entry["text"],
                                   "token_ids": {"path": f"tokens/{name}.bin", "sha256": entry["expected_token_ids_sha256"], "bytes": len(binary_bytes)},
                                   "index": {"path": f"tokens/{name}.index.npy", "sha256": entry["expected_index_sha256"]}})
        shared = {"tokenizer_sha256": cache.sha256(self.tokenizer_dir / "tokenizer.json"),
                  "document_orders": {"train": {"path": "orders/train.npy", "sha256": cache.sha256(self.source / "orders/train.npy")}},
                  "files": entries}
        (self.source / "dataset_manifest.json").write_text(json.dumps(shared, indent=2) + "\n")
        self.expected_manifest = (json.dumps({**shared, "files": expected_files, "complete": True}, indent=2) + "\n").encode()
        (self.source / "manifest.json").write_bytes(self.expected_manifest)
        (self.source / "verification.json").write_text(json.dumps({
            "passed": True, "dataset_manifest_sha256": cache.sha256(self.source / "dataset_manifest.json"),
            "manifest_sha256": hashlib.sha256(self.expected_manifest).hexdigest()}) + "\n")
        self.initial_order = (self.source / "orders/train.npy").read_bytes()

    def assert_exact_cache(self, root):
        for name, expected in self.expected.items():
            self.assertEqual((root / "tokens" / name).read_bytes(), expected)
        self.assertEqual((root / "manifest.json").read_bytes(), self.expected_manifest)
        self.assertEqual((root / "orders/train.npy").read_bytes(), self.initial_order)

    def test_sequential_and_parallel_match_independent_oracle(self):
        parallel = self.base / "parallel"
        shutil.copytree(self.source, parallel)
        with patch.dict(os.environ, {"RAYON_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "true"}):
            sequential_result = cache.rebuild(self.source, self.tokenizer_dir)
            parallel_result = cache.rebuild(parallel, self.tokenizer_dir, workers=2)
        self.assertTrue(sequential_result["complete"])
        self.assertTrue(parallel_result["complete"])
        self.assert_exact_cache(self.source)
        self.assert_exact_cache(parallel)

    def test_resume_reuses_only_checked_shards_and_repairs_interruption(self):
        partial = cache.rebuild(self.source, self.tokenizer_dir, limit_shards=1)
        self.assertFalse(partial["complete"])
        first = self.source / "tokens/train-00000.bin"
        first_stat = first.stat().st_mtime_ns
        (self.source / "tokens/train-00001.partial.bin").write_bytes(b"interrupted")
        resumed = cache.rebuild(self.source, self.tokenizer_dir, resume=True)
        self.assertEqual([s["reused_checked_cache"] for s in resumed["shards"]], [True, False])
        self.assertEqual(first.stat().st_mtime_ns, first_stat)
        self.assert_exact_cache(self.source)
        first.write_bytes(b"corrupt")
        repaired = cache.rebuild(self.source, self.tokenizer_dir, resume=True)
        self.assertEqual([s["reused_checked_cache"] for s in repaired["shards"]], [False, True])
        self.assert_exact_cache(self.source)
        (self.source / "tokens/train-00001.index.npy").unlink()
        missing_index = cache.rebuild(self.source, self.tokenizer_dir, resume=True)
        self.assertFalse(missing_index["shards"][1]["reused_checked_cache"])
        self.assert_exact_cache(self.source)

    def test_resume_still_rejects_changed_source_and_order(self):
        cache.rebuild(self.source, self.tokenizer_dir)
        text = self.source / "data/train-00000.parquet"
        original = text.read_bytes()
        text.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(ValueError, "Text Parquet checksum mismatch"):
            cache.rebuild(self.source, self.tokenizer_dir, resume=True)
        text.write_bytes(original)
        (self.source / "orders/train.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Document order checksum mismatch"):
            cache.rebuild(self.source, self.tokenizer_dir, resume=True)


if __name__ == "__main__":
    unittest.main()
