"""Check ordered stream coverage, rank ownership, recovery, and frozen sampling."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torchtitan.components.loss import IGNORE_INDEX

from tests.test_document_packing import fixture
from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset


class FixedDocumentOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs, self.tokenizer = fixture(self.root)
        self.order = [4, 1, 3, 0, 2]
        path = self.root / "order.npy"
        np.save(path, np.array(self.order, dtype=np.uint32))
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.manifest["document_orders"] = {
            split: {"path": "order.npy", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for split in ("train", "validation")
        }
        self.write_manifest()

    def write_manifest(self):
        data = json.dumps(self.manifest).encode()
        (self.root / "manifest.json").write_bytes(data)
        (self.root / "verification.json").write_text(json.dumps({
            "passed": True, "manifest_sha256": hashlib.sha256(data).hexdigest()}))

    def dataset(self, **kwargs):
        return PackedDocumentDataset(self.root, split="train", seq_len=16, **kwargs)

    def test_global_shuffle_precedes_rank_split_and_preserves_targets(self):
        for dp in (1, 2, 3):
            targets = []
            for rank in range(dp):
                docs = [self.docs[i] for i in self.order[rank::dp]]
                stream = sum(([254] + d + [255] for d in docs), [])
                expected = [(a, b) for a, b in zip(stream, stream[1:]) if b != 254]
                ds, observed = self.dataset(dp_world_size=dp, dp_rank=rank), []
                it = iter(ds)
                for step in range(100):
                    ds.seq_len = (8, 24, 16)[step % 3]
                    try:
                        inputs, labels = next(it)
                    except StopIteration:
                        break
                    observed.extend((int(a), int(b)) for a, b in zip(inputs["input"], labels)
                                    if b != IGNORE_INDEX)
                else:
                    self.fail("Finite ordered stream did not terminate")
                self.assertEqual(observed, expected)
                self.assertEqual(ds.epoch, 0)
                self.assertEqual(ds.dropped_tail_tokens, 0)
                targets.extend(b for _, b in observed)
            self.assertEqual(sorted(targets), sorted(sum((d + [255] for d in self.docs), [])))

    def test_checkpoint_replays_across_order_boundaries_and_epochs(self):
        ds = self.dataset(infinite=True)
        it = iter(ds)
        for step in range(12):
            ds.seq_len = (8, 24, 16)[step % 3]
            next(it)
            state = copy.deepcopy(ds.state_dict())
            expected = next(it)
            restored = self.dataset(infinite=True)
            restored.load_state_dict(state)
            restored.seq_len = ds.seq_len
            actual = next(iter(restored))
            for key in expected[0]:
                torch.testing.assert_close(actual[0][key], expected[0][key], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_rejects_duplicate_or_modified_order(self):
        np.save(self.root / "order.npy", np.array([4, 1, 3, 0, 0], dtype=np.uint32))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.dataset()
        for entry in self.manifest["document_orders"].values():
            entry["sha256"] = hashlib.sha256((self.root / "order.npy").read_bytes()).hexdigest()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.dataset()

if __name__ == "__main__":
    unittest.main()
