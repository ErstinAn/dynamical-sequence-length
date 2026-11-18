"""Check stream coverage, recovery, boundaries, and validation budgets."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchtitan.components.loss import IGNORE_INDEX

from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset


def fixture(root):
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text("{}")
    docs = [list(range(1, 4)), list(range(10, 18)), list(range(30, 55)), [70], list(range(80, 96))]
    entries = []
    for split in ("train", "validation"):
        for shard, group in enumerate((docs[:2], docs[2:])):
            tokens, index = [], []
            for ids in group:
                index.append([len(tokens), len(ids)])
                tokens.extend(ids)
            name = f"{split}-{shard}"
            np.array(tokens, dtype="<u4").tofile(root / f"{name}.bin")
            np.save(root / f"{name}.npy", np.array(index))
            entries.append(dict(split=split, index={"path": f"{name}.npy"},
                                token_ids={"path": f"{name}.bin"}, text={"path": f"{name}.parquet"}))
    manifest = dict(complete=True, files=entries, selection={"minimum_llama3_tokens": 1},
                    tokenizer_sha256=hashlib.sha256(b"{}").hexdigest(),
                    tokenizer={"vocab_size": 256, "bos_id": 254, "eos_id": 255})
    encoded = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(encoded)
    (root / "verification.json").write_text(json.dumps(dict(passed=True, manifest_sha256=hashlib.sha256(encoded).hexdigest())))
    return docs, tokenizer


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs, self.tokenizer = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return PackedDocumentDataset(self.root, split="train", seq_len=16, **kwargs)

    def test_exact_coverage_and_parallel_assignment(self):
        for dp in (1, 2, 3):
            all_targets = []
            for rank in range(dp):
                selected = self.docs[rank::dp]
                stream = sum(([254] + d + [255] for d in selected), [])
                expected = [(a, b) for a, b in zip(stream, stream[1:]) if b != 254]
                ds = self.dataset(dp_world_size=dp, dp_rank=rank)
                observed = []
                for inputs, labels in ds:
                    observed += [(int(a), int(b)) for a, b in zip(inputs["input"], labels) if b != IGNORE_INDEX]
                    for pos in (inputs["input"] == 254).nonzero(as_tuple=True)[0]:
                        self.assertEqual(int(inputs["positions"][pos]), 0)
                        if pos > 0:
                            self.assertEqual(int(labels[pos - 1]), IGNORE_INDEX)
                    self.assertEqual(int(inputs["positions"][0]), 0)
                    self.assertLess(int(inputs["positions"].max()), 16)
                    self.assertTrue(ds.last_fragments)
                self.assertEqual(observed, expected)
                self.assertEqual(ds.dropped_tail_tokens, 0)
                all_targets += [b for _, b in observed]
            self.assertEqual(sorted(all_targets), sorted(sum((d + [255] for d in self.docs), [])))

    def test_recovery_after_shards_epochs_and_shape_changes(self):
        ds = self.dataset(infinite=True)
        iterator = iter(ds)
        for i in range(15):
            next(iterator)
            saved = copy.deepcopy(ds.state_dict())
            ds.seq_len = 8 if i % 2 else 24
            expected = next(iterator)
            restored = self.dataset(infinite=True)
            restored.seq_len = ds.seq_len
            restored.load_state_dict(saved)
            actual = next(iter(restored))
            for key in expected[0]:
                torch.testing.assert_close(expected[0][key], actual[0][key])
            torch.testing.assert_close(expected[1], actual[1])
        self.assertGreater(ds.epoch, 0)
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.dataset(infinite=True).load_state_dict({**saved, "packing_protocol": "old"})
        with self.assertRaisesRegex(ValueError, "split"):
            PackedDocumentDataset(self.root, split="validation", seq_len=16, infinite=True).load_state_dict(saved)

    def test_batch_loader_checkpoint_with_changed_shape(self):
        from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentBatchDataLoader
        tokenizer = SimpleNamespace(tokenizer_path=str(self.tokenizer), get_vocab_size=lambda: 256)
        config = LongDocumentBatchDataLoader.Config(dataset_path=str(self.root), packing="stream", infinite=True)
        def build():
            return config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                                seq_len=32, local_batch_size=2)
        loader = build()
        iterator = iter(loader)
        next(iterator)
        state = copy.deepcopy(loader.state_dict())
        loader.set_batch_shape(seq_len=24, microbatch_size=3)
        expected = next(iterator)
        restored = build()
        restored.load_state_dict(state)
        restored.set_batch_shape(seq_len=24, microbatch_size=3)
        actual = next(iter(restored))
        for key in expected[0]:
            torch.testing.assert_close(expected[0][key], actual[0][key])
        torch.testing.assert_close(expected[1], actual[1])


if __name__ == "__main__":
    unittest.main()
