"""Numerical, sampling, state and event contracts for the independent X monitor."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
import yaml

from tests.test_grad_var_trainer import batches, tiny_trainer
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataset
from torchtitan_dynamic_seq_len.grad_var.metrics import XWandBLogger
from torchtitan_dynamic_seq_len.grad_var.monitor import GradVarMonitorConfig, XMonitor
from torchtitan_dynamic_seq_len.grad_var.probes import compute_group_gradient, maybe_monitor
from torchtitan_dynamic_seq_len.grad_var.sampler import ParentWindowSampler


class SyntheticSampler:
    def sample_groups(self, completed_updates, num_groups, group_sequences):
        return np.arange(num_groups * group_sequences).reshape(num_groups, group_sequences)

    def microbatches(self, anchors, length, dp_rank, dp_degree, local_batch_size):
        for ids in torch.tensor(anchors[dp_rank::dp_degree]).split(local_batch_size):
            values = torch.stack(
                [torch.arange(length, dtype=torch.float64) + int(i) / 13 for i in ids]
            )
            yield {"input": values}, values.sin()


def x_trainer(model=None, interval=2):
    trainer = tiny_trainer(False, k=2, b=2, seq=3, log_freq=3, model=model)
    cfg = GradVarMonitorConfig(
        enable=True,
        mode="x",
        interval=interval,
        lengths=[3],
        fit_lengths=[3],
        held_out_lengths=[],
        parent_length=8,
        num_groups=3,
        group_sequences=4,
        local_batch_size=2,
    )
    trainer.config.grad_var_monitor = cfg
    trainer.grad_var_monitor = XMonitor(trainer.model_parts[0].named_parameters(), cfg.num_groups)
    trainer.probe_loader = SyntheticSampler()
    trainer._x_dp_rank, trainer._x_dp_degree = 0, 1
    trainer._x_last_event, trainer._x_signature = -1, "fixture"
    return trainer


class XMonitorTests(unittest.TestCase):
    def test_arithmetic_missing_tied_and_two_group_identity(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        monitor = XMonitor([("p", p), ("tied", p)], 3)
        monitor.begin()
        for sample in [(1, 2), (3, 0), (2, 4)]:
            p.grad = torch.tensor(sample, dtype=torch.float64)
            monitor.observe_group_gradient()
        self.assertEqual(monitor.finish(32), 160)
        self.assertEqual(96 * (5 / 3), 160)
        torch.testing.assert_close(monitor.mean["p"], torch.tensor([2.0, 2.0], dtype=torch.float64))
        m = XMonitor([("p", p)], 2)
        m.begin()
        p.grad = torch.tensor([1.0, 2.0], dtype=torch.float64)
        m.observe_group_gradient()
        p.grad = None
        m.observe_group_gradient()
        self.assertEqual(m.finish(32), 32 / 2 * 5)
        m.begin()
        p.grad = torch.full_like(p, float("nan"))
        m.observe_group_gradient()
        p.grad = None
        m.observe_group_gradient()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.finish(32)

    def test_config_guards(self):
        cfg = GradVarMonitorConfig(mode="x")
        for degree in (1, 2, 4):
            cfg.validate(degree)
        for kwargs in (
            {"num_groups": 1},
            {"interval": 0},
            {"lengths": []},
            {"lengths": [64, 64]},
            {"parent_length": 1024},
            {"local_batch_size": 31},
            {"replicate_groups": 1},
            {"replicate_groups": 3},
            {"replicate_groups": 8},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GradVarMonitorConfig(mode="x", **kwargs).validate(4)

    def test_independent_replicates_match_direct_global_gradients(self):
        trainer = x_trainer()
        cfg = trainer.config.grad_var_monitor
        cfg.num_groups, cfg.replicate_groups = 8, 2
        model = trainer.model_parts[0]
        trainer.grad_var_monitor = XMonitor(model.named_parameters(), cfg.num_groups)
        trainer._x_replicate_monitor = XMonitor(model.named_parameters(), cfg.replicate_groups)
        emitted = []
        trainer.metrics_processor.emit_probe_metrics = lambda p: emitted.append(dict(p))
        gradients = []
        for anchors in trainer.probe_loader.sample_groups(0, 8, 4):
            inputs, labels = next(trainer.probe_loader.microbatches(anchors, 3, 0, 1, 4))
            grads = torch.autograd.grad(
                (model(inputs["input"]) - labels).square().mean(), tuple(model.parameters())
            )
            gradients.append(torch.cat([g.flatten() for g in grads]))
        gradients = torch.stack(gradients)

        def reference(rows):
            return (12 * (rows - rows.mean(0)).square().sum() / (len(rows) - 1)).item()

        with tempfile.TemporaryDirectory() as folder:
            trainer.config.dump_folder = folder
            before = copy.deepcopy(model.state_dict())
            maybe_monitor(trainer, 0)
            event = json.loads((Path(folder) / "x_replicates.jsonl").read_text())
            self.assertEqual(event["groups_per_replicate"], 2)
            self.assertEqual(event["group_sequences"], 4)
            self.assertEqual(event["step"], 0)
            self.assertEqual(len(event["replicates"]), 4)
            self.assertAlmostEqual(emitted[0]["grad_var/x_s3"], reference(gradients), places=10)
            for i, replicate in enumerate(event["replicates"]):
                self.assertEqual(set(replicate), {"grad_var/x_s3"})
                self.assertAlmostEqual(
                    replicate["grad_var/x_s3"], reference(gradients[2 * i : 2 * i + 2]), places=10
                )
            self.assertEqual(set(emitted[0]), {"grad_var/step", "grad_var/x_s3"})
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            self.assertEqual(trainer._x_replicate_monitor.count, 0)

    def test_pipeline_accepts_configured_grid_and_checks_repeats(self):
        from scripts.run_x_monitor_pipeline import validate_local_artifacts

        config_path = (
            Path(__file__).resolve().parents[1] / "legacy/configs/grad_var/x_monitor_expanded_smoke.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        fixed = config["fixed"]
        lengths = fixed["grad_var_monitor.lengths"]
        values = {f"grad_var/x_s{s}": float(s) for s in lengths}
        with tempfile.TemporaryDirectory() as folder:
            stage = {"config": str(config_path), "dump_folder": folder}
            events = [{"metrics": {"grad_var/step": step, **values}} for step in (0, 2, 4)]
            repeats = [
                {
                    "step": step,
                    "group_sequences": 128,
                    "groups_per_replicate": 8,
                    "replicates": [dict(values) for _ in range(4)],
                }
                for step in (0, 2, 4)
            ]
            (Path(folder) / "x_events.jsonl").write_text("\n".join(map(json.dumps, events)))
            repeat_file = Path(folder) / "x_replicates.jsonl"
            repeat_file.write_text("\n".join(map(json.dumps, repeats)))
            result, _, keys, steps = validate_local_artifacts(stage)
            self.assertEqual(result, events)
            self.assertEqual(len(keys), 17)
            self.assertEqual(steps, [0, 2, 4])
            repeats[1]["replicates"][0].pop("grad_var/x_s1280")
            repeat_file.write_text("\n".join(map(json.dumps, repeats)))
            with self.assertRaisesRegex(RuntimeError, "replicate payload"):
                validate_local_artifacts(stage)

    def test_real_pool_prefix_alignment_and_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = object.__new__(LongDocumentDataset)
            dataset.root = Path(folder)
            # A seven-token document has eight labels incl EOS; a 16-token
            # document has two full 8-target windows and one discarded label.
            raw = np.arange(23, dtype="<u4")
            raw.tofile(dataset.root / "tokens.bin")
            dataset.indexes = [np.array([[0, 7], [7, 16]])]
            dataset.files = [{"token_ids": {"path": "tokens.bin"}}]
            dataset.bos, dataset.eos = 101, 102
            sampler = ParentWindowSampler(dataset, 8, 0)
            self.assertEqual(sampler.pool_size, 3)
            inputs, labels = sampler.read(0, 8)
            self.assertEqual(inputs["input"].tolist(), [101, 0, 1, 2, 3, 4, 5, 6])
            self.assertEqual(labels.tolist(), [0, 1, 2, 3, 4, 5, 6, 102])
            short, short_labels = sampler.read(2, 3)
            long, long_labels = sampler.read(2, 8)
            torch.testing.assert_close(short["input"], long["input"][:3])
            torch.testing.assert_close(short_labels, long_labels[:3])
            self.assertEqual(short["positions"].tolist(), [0, 1, 2])
            rng = np.random.get_state()
            anchors = sampler.sample_groups(0, 3, 128)
            np.testing.assert_array_equal(anchors, sampler.sample_groups(0, 3, 128))
            self.assertFalse(np.array_equal(anchors, sampler.sample_groups(2, 3, 128)))
            np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
            self.assertEqual(sum(len(anchors[0][r::4]) for r in range(4)), 128)

    def test_accumulation_matches_direct_gradient(self):
        trainer = x_trainer()
        for batch_size in (1, 2, 4):
            trainer.config.grad_var_monitor.local_batch_size = batch_size
            trainer.grad_var_monitor.begin()
            compute_group_gradient(trainer, np.arange(4), 3)
            actual = [p.grad.clone() for p in trainer.model_parts[0].parameters()]
            inputs, labels = next(SyntheticSampler().microbatches(np.arange(4), 3, 0, 1, 4))
            reference = torch.autograd.grad(
                (trainer.model_parts[0](inputs["input"]) - labels).square().mean(),
                tuple(trainer.model_parts[0].parameters()),
            )
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)

    def test_events_training_equivalence_resume_and_exception_restore(self):
        off = tiny_trainer(False, k=2, b=2, seq=3, log_freq=3)
        on = x_trainer(copy.deepcopy(off.model_parts[0]))
        # Mutable buffers and dropout exercise restoration across groups/events.
        events = []
        on.metrics_processor.emit_probe_metrics = lambda p: events.append(dict(p))
        data = batches(k=2, b=2, seq=3)
        rng = torch.get_rng_state().clone()
        for step in range(1, 5):
            for t in (off, on):
                t.step = step
                t.metrics_processor.data_loading_times = [0.001]
                t.train_step(iter(data))
        self.assertEqual([p["grad_var/step"] for p in events], [0, 2, 4])
        self.assertEqual(set(events[0]), {"grad_var/step", "grad_var/x_s3"})
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertEqual(off.ntokens_seen, on.ntokens_seen)
        for a, b in zip(off.model_parts[0].parameters(), on.model_parts[0].parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(
            off.optimizers.optimizer.state.values(), on.optimizers.optimizer.state.values()
        ):
            for k in a:
                torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
        self.assertEqual(
            off.lr_schedulers.scheduler.state_dict(), on.lr_schedulers.scheduler.state_dict()
        )
        state = on.state_dict()
        on.load_state_dict(state)
        maybe_monitor(on, 4)
        self.assertEqual(len(events), 3)
        model = on.model_parts[0]
        model.register_buffer("counter", torch.tensor(0.0))

        def fail(*args, **kwargs):
            model.counter.add_(1)
            torch.rand(10)
            raise RuntimeError("injected")

        on.probe_loader.microbatches = fail
        with self.assertRaisesRegex(RuntimeError, "injected"):
            maybe_monitor(on, 6)
        self.assertEqual(model.counter.item(), 0)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertFalse(on._x_probe_active)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_dropout_buffers_and_training_stream_are_preserved(self):
        class StochasticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3).double()
                self.register_buffer("calls", torch.zeros(()))

            def forward(self, value):
                self.calls.add_(1)
                return self.linear(torch.nn.functional.dropout(value, p=0.2, training=True))

        model = StochasticModel()
        off = tiny_trainer(False, k=2, b=2, seq=3, model=copy.deepcopy(model))
        on = x_trainer(copy.deepcopy(model))
        on.metrics_processor.emit_probe_metrics = Mock()
        data = batches(k=2, b=2, seq=3)
        for step in range(1, 3):
            before = torch.get_rng_state()
            off.step = on.step = step
            off.metrics_processor.data_loading_times = [0.001]
            on.metrics_processor.data_loading_times = [0.001]
            off.train_step(iter(data))
            after = torch.get_rng_state()
            torch.set_rng_state(before)
            iterator = iter(data)
            on.train_step(iterator)
            self.assertIsNone(next(iterator, None))
            torch.testing.assert_close(torch.get_rng_state(), after, atol=0, rtol=0)
            for key, value in off.model_parts[0].state_dict().items():
                torch.testing.assert_close(
                    value, on.model_parts[0].state_dict()[key], atol=0, rtol=0
                )

    def test_probe_padding_guard_and_initial_disabled(self):
        trainer = x_trainer()
        trainer.config.grad_var_monitor.include_initial = False
        trainer.metrics_processor.emit_probe_metrics = Mock()
        maybe_monitor(trainer, 0)
        trainer.metrics_processor.emit_probe_metrics.assert_not_called()
        original = trainer.probe_loader.microbatches

        def padded(*args, **kwargs):
            for inputs, labels in original(*args, **kwargs):
                labels[0, 0] = -100
                yield inputs, labels

        trainer.probe_loader.microbatches = padded
        with self.assertRaisesRegex(RuntimeError, "valid labels"):
            maybe_monitor(trainer, 2)
        trainer.metrics_processor.emit_probe_metrics.assert_not_called()

    def test_validation_requires_explicit_matching_artifact(self):
        import json

        from torchtitan_dynamic_seq_len.config_registry import llama3_100m_long_documents_x_monitor
        from torchtitan_dynamic_seq_len.grad_var.probes import validate_config

        config = llama3_100m_long_documents_x_monitor()
        config.validator.enable = True
        config.validator.token_budget = 8192
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_config(config)
        with tempfile.TemporaryDirectory() as folder:
            config.validator.artifact_path = folder
            config.dataloader.expected_manifest_sha256 = "expected"
            manifest = {
                "sequence_length": 2048,
                "token_budget": 8192,
                "source": {"split": "validation", "manifest_sha256": "expected"},
            }
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "length/token budget"):
                validate_config(config)
            manifest["sequence_length"] = 1024
            manifest["source"]["split"] = "train"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "validation split"):
                validate_config(config)
            manifest["source"]["split"] = "validation"
            path.write_text(json.dumps(manifest))
            validate_config(config)

    def test_wandb_commit_contract(self):
        wrapped = Mock()
        wrapped.tag = None
        logger = XWandBLogger(wrapped, {"total_sequences": 1024})
        logger.log({"grad_var/step": 0}, 0)
        logger.log({"train/loss": 1}, 2)
        logger.log({"grad_var/step": 2, "grad_var/x_s64": 3}, 2)
        self.assertTrue(all(c.kwargs["commit"] is False for c in wrapped.wandb.log.call_args_list))
        wrapped.wandb.run.define_metric.assert_any_call(
            "grad_var/x_s*", step_metric="grad_var/step"
        )
        with self.assertRaises(RuntimeError):
            logger.log({}, 1)
        logger.close()
        wrapped.wandb.log.assert_called_with({}, commit=True)


if __name__ == "__main__":
    unittest.main()
