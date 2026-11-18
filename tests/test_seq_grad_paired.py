"""Shared parents, random-window alignment, DP invariance and full-H gradients."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from tests.test_document_packing import fixture
from tests.test_seq_grad import explicit_gradient, probe_trainer
from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset
from torchtitan_dynamic_seq_len.seq_grad.sampler import (
    PackedProbeLoader, PairedWindowProbeLoader, SingleDocumentProbeLoader, stream_seed,
)


def flatten(batches):
    return ({name: torch.cat([x[name] for x, _ in batches]) for name in batches[0][0]},
            torch.cat([y for _, y in batches]))


class PairedSamplerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        fixture(root)
        self.source = PackedDocumentDataset(root, split="train", seq_len=16, infinite=True)
        next(iter(self.source))
        self.state = copy.deepcopy(self.source.state_dict())
        self.base = PackedProbeLoader(self.source)
        self.base.sample_batch = Mock(wraps=self.base.sample_batch)
        self.sampler = PairedWindowProbeLoader(self.base, 64)
        self.kw = dict(global_sequences=16, local_batch_size=2, rank=0, degree=1, seed=3, step=7)

    def tearDown(self):
        self.temp.cleanup()

    def test_one_parent_batch_aligned_random_crops_and_document_resets(self):
        parents = self.sampler.sample_batch(length=64, **self.kw)
        x, y = flatten(parents)
        saved = copy.deepcopy(parents)
        used_offsets = []
        for length in (8, 16, 32):
            cropped_x, cropped_y = flatten(self.sampler.sample_batch(length=length, **self.kw))
            for i in range(16):
                offset = int(np.random.default_rng(stream_seed(3, 7, length, i, "paired-window-v1")).integers(65-length))
                used_offsets.append(offset)
                torch.testing.assert_close(cropped_x["input"][i], x["input"][i, offset:offset+length], rtol=0, atol=0)
                torch.testing.assert_close(cropped_y[i], y[i, offset:offset+length], rtol=0, atol=0)
                # Build positions independently from document-start markers.
                original = x["positions"][i, offset:offset+length].tolist()
                expected = []
                for j, position in enumerate(original):
                    expected.append(0 if j == 0 or position == 0 else expected[-1] + 1)
                self.assertEqual(cropped_x["positions"][i].tolist(), expected)
        self.assertGreater(len(set(used_offsets)), 10)
        self.assertEqual(self.base.sample_batch.call_count, 1)
        for (a, b), (c, d) in zip(parents, saved):
            for name in a:
                torch.testing.assert_close(a[name], c[name], rtol=0, atol=0)
            torch.testing.assert_close(b, d, rtol=0, atol=0)
        self.assertEqual(self.source.state_dict(), self.state)
        # H is the whole parent including every original valid/ignored target.
        same_x, same_y = flatten(self.sampler.sample_batch(length=64, **self.kw))
        torch.testing.assert_close(same_y, y, rtol=0, atol=0)
        torch.testing.assert_close(same_x["positions"], x["positions"], rtol=0, atol=0)
        self.sampler.sample_batch(length=8, **{**self.kw, "step": 8})
        self.assertEqual(self.base.sample_batch.call_count, 2)

    def test_dp_microbatch_order_and_replay_invariance(self):
        expected_x, expected_y = flatten(self.sampler.sample_batch(length=16, **self.kw))
        for degree in (1, 2, 4):
            for batch_size in (1, 2, 4):
                for rank in range(degree):
                    self.sampler.clear_event()
                    # Different candidate call order must retain the same parents/windows.
                    args = {**self.kw, "rank": rank, "degree": degree, "local_batch_size": batch_size}
                    self.sampler.sample_batch(length=32, **args)
                    actual_x, actual_y = flatten(self.sampler.sample_batch(length=16, **args))
                    torch.testing.assert_close(actual_y, expected_y[rank::degree], rtol=0, atol=0)
                    for name in actual_x:
                        torch.testing.assert_close(actual_x[name], expected_x[name][rank::degree], rtol=0, atol=0)
        self.assertEqual(self.source.state_dict(), self.state)

    def test_single_document_parents_exclude_short_records_and_boundary_tokens(self):
        base = SingleDocumentProbeLoader(self.source, 16)
        self.assertEqual(base.identity["eligible_documents"], 1)  # Only the 25-token fixture doc is >16.
        self.assertEqual(base.identity["population_parent_starts"], 9)
        sampler = PairedWindowProbeLoader(base, 16)
        parents = sampler.sample_batch(length=16, **self.kw)
        parent_x, parent_y = flatten(parents)
        self.assertTrue(torch.all((parent_x["input"] >= 30) & (parent_x["input"] <= 53)))
        torch.testing.assert_close(parent_y, parent_x["input"] + 1, rtol=0, atol=0)
        torch.testing.assert_close(parent_x["positions"], torch.arange(16).expand(16,16), rtol=0, atol=0)
        for rank in range(4):
            _, y = flatten(sampler.sample_batch(length=16, **{**self.kw, "degree":4, "rank":rank}))
            torch.testing.assert_close(y, parent_y[rank::4], rtol=0, atol=0)
        x, y = flatten(sampler.sample_batch(length=8, **self.kw))
        torch.testing.assert_close(y, x["input"] + 1, rtol=0, atol=0)
        torch.testing.assert_close(x["positions"], torch.arange(8).expand(16,8), rtol=0, atol=0)
        self.assertEqual(self.source.state_dict(), self.state)
        with self.assertRaisesRegex(ValueError, "No documents"):
            SingleDocumentProbeLoader(self.source, 25)


class PairedGradientTests(unittest.TestCase):
    def test_monitor_attention_hook_preserves_native_bookkeeping_and_training_path(self):
        from tests.seq_grad_single_document_gpu import attention_trainer
        from torchtitan_dynamic_seq_len.config_registry import _model
        model = _model("flex", dim=64, num_layers=2, num_heads=4, num_kv_heads=2, vocab_size=128).build()
        trainer = attention_trainer(model)
        inputs = {"input": torch.ones(2, 8, dtype=torch.long), "positions": torch.arange(8).expand(2, 8)}
        labels = inputs["input"].clone()
        with patch("torchtitan_dynamic_seq_len.seq_grad.trainer.create_attention_mask", return_value="causal") as causal, \
             patch.object(model, "get_attention_masks", return_value="packed") as packed:
            _, _, kwargs = trainer.post_dataloading_process(inputs, labels)
            self.assertEqual(kwargs["attention_masks"], "causal")
            self.assertEqual(trainer.ntokens_seen, 16)
            causal.assert_called_once()
            packed.assert_not_called()
            trainer._probe_active = False
            _, _, kwargs = trainer.post_dataloading_process(inputs, labels)
            self.assertEqual(kwargs["attention_masks"], "packed")
            self.assertEqual(trainer.ntokens_seen, 32)
            packed.assert_called_once()

    def trainer(self):
        trainer = probe_trainer()
        trainer.config.seq_grad.sampling = "paired_window"
        trainer.config.seq_grad.lengths.append(16)  # e(H) must be zero.
        trainer.probe_loader = PairedWindowProbeLoader(trainer.probe_loader, 16)
        return trainer

    def test_distance_to_full_parent_token_mean_and_cache_release(self):
        reference = self.trainer()
        direct = {}
        for length in (16, 2, 4, 8):
            data = reference.probe_loader.sample_batch(length=length, global_sequences=8,
                local_batch_size=8, rank=0, degree=1, seed=0, step=0)
            direct[length] = explicit_gradient(copy.deepcopy(reference.model_parts[0]), data)
        for microbatch in (1, 2, 4, 8):
            trainer = self.trainer()
            trainer.config.seq_grad.probe_local_batch_size = microbatch
            before = copy.deepcopy(trainer.model_parts[0].state_dict())
            trainer.maybe_monitor(0)
            metrics = trainer.metrics_processor.logger.log.call_args.args[0]
            for length in (2, 4, 8, 16):
                self.assertAlmostEqual(metrics[f"context_bias/distance_s{length}"],
                    (direct[length] - direct[16]).norm().item(), places=12)
            self.assertEqual(metrics["context_bias/distance_s16"], 0)
            self.assertIsNone(trainer.probe_loader._parents)
            self.assertEqual(trainer.ntokens_seen, 0)
            for name, value in trainer.model_parts[0].state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_failure_releases_parent_cache_and_gradients(self):
        trainer = self.trainer()
        trainer._statistics = Mock(side_effect=RuntimeError("injected failure"))
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            trainer.maybe_monitor(0)
        self.assertIsNone(trainer.probe_loader._parents)
        self.assertEqual(trainer._last_event, -1)
        self.assertTrue(all(p.grad is None for p in trainer.model_parts[0].parameters()))


if __name__ == "__main__":
    unittest.main()
