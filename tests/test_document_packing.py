"""Check stream coverage, recovery, boundaries, and validation budgets."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchtitan.components.loss import IGNORE_INDEX

from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset
from torchtitan_dynamic_seq_len.data.seq_grad_windows import crop_batches
from torchtitan_dynamic_seq_len.data.fixed_validation import FrozenValidationArtifact
from torchtitan_dynamic_seq_len.loss_grad_seq.attention import attention_positions
from torchtitan_dynamic_seq_len.loss_grad_seq.validation import ValidationPrefixView
from scripts.prepare_packed_validation import prepare_packed


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

    def test_crop_positions_and_mask_toggle(self):
        batch = next(iter(self.dataset()))
        batches = [({k: v[None] for k, v in batch[0].items()}, batch[1][None])]
        original = copy.deepcopy(batches)
        for step in range(20):
            crop, labels = crop_batches(batches, 8, seed=1, step=step, rank=0, stream="test")[0]
            pos = crop["positions"]
            self.assertEqual(int(pos[0, 0]), 0)
            torch.testing.assert_close(attention_positions(pos, True), pos)
            torch.testing.assert_close(attention_positions(pos, False), torch.arange(8)[None])
            # Every BOS begins a new segment, including those inside the crop.
            self.assertTrue(torch.all(pos[crop["input"] == 254] == 0))
        torch.testing.assert_close(batches[0][0]["positions"], original[0][0]["positions"])

    def test_validation_budget_prefix_and_provenance(self):
        output = self.root / "frozen"
        manifest = prepare_packed(self.root, output, self.tokenizer, length=16, count=4)
        artifact = FrozenValidationArtifact(output, token_budget=manifest["token_budget"],
                                            sequence_length=16, tokenizer_path=self.tokenizer)
        self.assertLess(artifact.token_budget, 64)
        self.assertEqual(artifact.token_budget, sum(len(d) + 1 for d in self.docs))
        prefix = ValidationPrefixView(artifact, 8)
        self.assertEqual(prefix.token_budget, int(np.count_nonzero(artifact.arrays["labels"][:, :8] != IGNORE_INDEX)))
        provenance = json.loads((output / "documents.json").read_text())
        self.assertGreater(len(provenance[0]["fragments"]), 1)
        with self.assertRaisesRegex(ValueError, "budget"):
            FrozenValidationArtifact(output, token_budget=64, sequence_length=16, tokenizer_path=self.tokenizer)

    def test_failed_preparation_leaves_no_artifact(self):
        output = self.root / "too-many"
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            prepare_packed(self.root, output, self.tokenizer, length=16, count=100)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.partial-*")))

    def test_configuration_defaults_and_all_schedules(self):
        from torchtitan.config import ConfigManager
        from torchtitan_dynamic_seq_len.run_config import resolve_point
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "legacy/configs/longseq_packed").glob("*.yaml"))
        self.assertEqual(len(paths), 6)
        for dp in (1, 2, 4, 8):
            with patch.dict(os.environ, {"WORLD_SIZE": str(dp)}):
                for path in paths:
                    raw, _, _, options = resolve_point(path, 0)
                    cfg = ConfigManager().parse_args([
                        "--module", "torchtitan_dynamic_seq_len.config_registry",
                        "--config", raw["torchtitan_config"], *options])
                    schedules = cfg.validate_loss_grad()
                    self.assertTrue(cfg.document_masking)
                    self.assertTrue(cfg.validator.document_masking)
                    self.assertEqual(cfg.training.seq_len, 22256)
                    self.assertEqual(cfg.validator.eval_seq_len, 22256)
                    for step in (1, 80, 160, 161, 240, 320, 3840):
                        plan = cfg.plan(schedules, step, dp)
                        self.assertTrue(2048 <= plan.grad_length <= 8192)
                        self.assertEqual(plan.grad_length % 8, 0)
                    cfg.document_masking = False
                    cfg.validate_loss_grad()
                    self.assertFalse(cfg.validator.document_masking)
                    cfg.validator.dataloader.packing = "none"
                    with self.assertRaisesRegex(ValueError, "same packing"):
                        cfg.validate_loss_grad()
                    cfg.validator.dataloader.packing = "stream"
                    cfg.dataloader.packing = "none"
                    with self.assertRaisesRegex(ValueError, "same packing"):
                        cfg.validate_loss_grad()

    def test_masking_change_rejects_checkpoint(self):
        from tests.test_loss_grad_seq import dual_trainer
        trainer = dual_trainer()
        state = copy.deepcopy(trainer.state_dict())
        trainer.config.document_masking = False
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            trainer.load_state_dict(state)

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
