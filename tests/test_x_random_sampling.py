"""Check document boundaries, independent RNG streams and per-length event draws."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.test_x_monitor import SyntheticSampler, x_trainer
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataset
from torchtitan_dynamic_seq_len.grad_var.monitor import GradVarMonitorConfig
from torchtitan_dynamic_seq_len.grad_var.probes import maybe_monitor
from torchtitan_dynamic_seq_len.grad_var.sampler import RandomWindowSampler


def fixture(root):
    dataset = object.__new__(LongDocumentDataset)
    dataset.root = Path(root)
    dataset.bos, dataset.eos = 101, 102
    dataset.indexes, dataset.files = [], []
    # The middle file has no valid starts at s=8. The final file includes
    # another short document before a long document, testing repeated counts.
    documents = [[list(range(7))], [list(range(10, 12))], [[20], list(range(30, 46))]]
    for file, docs in enumerate(documents):
        raw, index = [], []
        for doc in docs:
            index.append([len(raw), len(doc)])
            raw.extend(doc)
        path = f"tokens{file}.bin"
        np.array(raw, dtype="<u4").tofile(dataset.root / path)
        dataset.indexes.append(np.array(index))
        dataset.files.append({"token_ids": {"path": path}})
    return dataset, [doc for docs in documents for doc in docs]


class RandomWindowTests(unittest.TestCase):
    def test_every_valid_start_exactly_once_no_document_crossing(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, docs = fixture(folder)
            sampler = RandomWindowSampler(dataset, [3, 8], 1)
            for length in (3, 8):
                expected = []
                for doc in docs:
                    virtual = [dataset.bos, *doc, dataset.eos]
                    expected.extend(
                        virtual[start : start + length + 1]
                        for start in range(len(virtual) - length)
                    )
                self.assertEqual(sampler.pool_sizes[length], len(expected))
                for anchor, tokens in enumerate(expected):
                    inputs, labels = sampler.read(anchor, length)
                    self.assertEqual(inputs["input"].tolist(), tokens[:-1])
                    self.assertEqual(labels.tolist(), tokens[1:])
                    self.assertEqual(inputs["positions"].tolist(), list(range(length)))
                for bad in (-1, len(expected)):
                    with self.assertRaises(ValueError):
                        sampler.read(bad, length)
            with self.assertRaisesRegex(ValueError, "No complete"):
                RandomWindowSampler(dataset, [100], 1)

    def test_replay_independent_lengths_and_events_preserve_global_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, _ = fixture(folder)
            sampler = RandomWindowSampler(dataset, [3, 8], 1)
            reverse = RandomWindowSampler(dataset, [8, 3, 4], 1)
            numpy_before = copy.deepcopy(np.random.get_state())
            torch_before = torch.get_rng_state().clone()
            first = sampler.sample_groups(50, 32, 128, length=3)
            np.testing.assert_array_equal(first, reverse.sample_groups(50, 32, 128, length=3))
            self.assertFalse(np.array_equal(first, sampler.sample_groups(100, 32, 128, length=3)))
            self.assertFalse(np.array_equal(first, sampler.sample_groups(50, 32, 128, length=8)))
            # Even equal-size pools must have different streams for different s.
            sampler.pool_sizes[8] = sampler.pool_sizes[3]
            self.assertFalse(np.array_equal(first, sampler.sample_groups(50, 32, 128, length=8)))
            after = np.random.get_state()
            np.testing.assert_array_equal(numpy_before[1], after[1])
            self.assertEqual(numpy_before[2:], after[2:])
            torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)

    def test_dp_ownership_keeps_duplicate_draws(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, _ = fixture(folder)
            sampler = RandomWindowSampler(dataset, [3], 1)
            draws = np.array([0, 0, 7, 6, 7, 1, 1, 2])
            actual = {}
            for rank in range(4):
                batches = list(sampler.microbatches(draws, 3, rank, 4, 1))
                for index, (inputs, labels) in enumerate(batches):
                    actual[rank + index * 4] = (inputs["input"][0], labels[0])
            self.assertEqual(len(actual), len(draws))
            for index, draw in enumerate(draws):
                inputs, labels = sampler.read(draw, 3)
                torch.testing.assert_close(actual[index][0], inputs["input"])
                torch.testing.assert_close(actual[index][1], labels)

    def test_event_resamples_each_length_and_replays_after_grid_reorder(self):
        class RecordingSampler(SyntheticSampler):
            def __init__(self):
                self.calls = []

            def sample_groups(self, step, groups, sequences, *, length):
                self.calls.append((step, length))
                rng = np.random.default_rng(np.random.SeedSequence([step, length]))
                return rng.integers(100, size=(groups, sequences))

        class ScaleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

            def forward(self, value):
                return self.scale * value

        trainer = x_trainer(model=ScaleModel())
        cfg = trainer.config.grad_var_monitor
        cfg.sampling = "independent_random_window"
        cfg.lengths = [3, 5]
        sampler = trainer.probe_loader = RecordingSampler()
        emitted = []
        trainer.metrics_processor.emit_probe_metrics = lambda p: emitted.append(dict(p))
        before = copy.deepcopy(trainer.model_parts[0].state_dict())
        maybe_monitor(trainer, 0)
        self.assertEqual(sampler.calls, [(0, 3), (0, 5)])
        cfg.lengths.reverse()
        trainer._x_last_event = -1
        maybe_monitor(trainer, 0)
        self.assertEqual(emitted[0], emitted[1])
        for key, value in trainer.model_parts[0].state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        maybe_monitor(trainer, 2)
        self.assertEqual(sampler.calls[-2:], [(2, 5), (2, 3)])
        self.assertFalse(trainer._x_probe_active)
        self.assertTrue(all(p.grad is None for p in trainer.model_parts[0].parameters()))

    def test_config_does_not_apply_parent_limit_to_random_windows(self):
        GradVarMonitorConfig(
            mode="x", sampling="independent_random_window", parent_length=0
        ).validate(4)
        with self.assertRaisesRegex(ValueError, "sampling"):
            GradVarMonitorConfig(mode="x", sampling="unknown").validate(4)


if __name__ == "__main__":
    unittest.main()
