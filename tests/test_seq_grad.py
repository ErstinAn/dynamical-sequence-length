"""Check sequence probes against direct gradients and the native optimizer step."""

import copy
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.config import ConfigManager

from tests.test_dynamic_seq import ScalarModel
from tests.test_grad_var_trainer import cpu_metrics, tiny_trainer
from torchtitan_dynamic_seq_len.data.seq_grad_windows import crop_batches
from torchtitan_dynamic_seq_len.run_config import resolve_point
from torchtitan_dynamic_seq_len.seq_grad.metrics import SeqGradMetricsProcessor, SeqGradWandBLogger
from torchtitan_dynamic_seq_len.seq_grad.monitor import SeqGradConfig, SeqGradMonitor
from torchtitan_dynamic_seq_len.seq_grad.trainer import SeqGradTrainer


def parents(k=2, batch=2, capacity=16):
    rng = torch.Generator().manual_seed(200)
    return [
        (
            {"input": torch.randn(batch, capacity, generator=rng, dtype=torch.float64)},
            torch.randn(batch, capacity, generator=rng, dtype=torch.float64),
        )
        for _ in range(k)
    ]


def probe_trainer(enabled=True, model=None, k=2, batch=2):
    fixture = tiny_trainer(False, k=k, b=batch, seq=16, model=model or ScalarModel())
    trainer = object.__new__(SeqGradTrainer)
    trainer.__dict__.update(fixture.__dict__)
    trainer.config.seq_grad = SeqGradConfig(enable=enabled, train_length=4, lengths=[2, 4, 8])
    trainer.config.training.steps = 4
    trainer._probe_active = False
    trainer._training_calls = 0
    trainer.n_valid_tokens_seen = 0
    trainer.seq_grad_monitor = SeqGradMonitor(trainer.model_parts[0].named_parameters())
    trainer.flops_by_length = dict.fromkeys([2, 4, 8, 16], 10)
    metrics = object.__new__(SeqGradMetricsProcessor)
    metrics.__dict__.update(cpu_metrics(trainer.parallel_dims).__dict__)
    metrics.config = SeqGradMetricsProcessor.Config(log_freq=1, profile_freq=2)
    metrics.reset_window()
    trainer.metrics_processor = metrics
    return trainer


def explicit_gradient(model, data):
    total = sum(int((labels != -100).sum()) for _, labels in data)
    model.zero_grad()
    for inputs, labels in data:
        valid = labels != -100
        ((model(inputs["input"]) - labels)[valid].square().sum() / total).backward()
    return torch.cat([p.grad.detach().flatten().clone() for p in model.parameters()])


class RandomModel(ScalarModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.tensor(0))

    def forward(self, inputs, **kwargs):
        self.calls.add_(1)
        return super().forward(inputs, **kwargs) * torch.rand_like(inputs)


class WindowsTest(unittest.TestCase):
    def test_contiguous_aligned_windows_and_document_positions(self):
        data = [
            (
                {
                    "input": torch.arange(16).reshape(2, 8),
                    "positions": torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]] * 2),
                },
                torch.arange(16).reshape(2, 8) + 1,
            )
        ]
        seen = set()
        for step in range(1, 100):
            cropped = crop_batches(data, 4, seed=3, step=step, rank=0, stream="train")
            inputs, labels = cropped[0]
            torch.testing.assert_close(labels, inputs["input"] + 1)
            torch.testing.assert_close(
                inputs["input"].diff(dim=1), torch.ones(2, 3, dtype=torch.long)
            )
            for row in range(2):
                start = int(inputs["input"][row, 0]) - row * 8
                seen.add(start)
                positions = data[0][0]["positions"][row, start : start + 4].tolist()
                offset = positions[0]
                if offset:
                    for i, value in enumerate(positions):
                        if value == 0:
                            break
                        positions[i] -= offset
                self.assertEqual(inputs["positions"][row].tolist(), positions)
        self.assertEqual(seen, set(range(5)))

    def test_seed_independence_and_full_length_identity(self):
        data = parents()
        state = torch.get_rng_state()
        first = crop_batches(data, 4, seed=1, step=2, rank=0, stream="train")
        crop_batches(data, 8, seed=1, step=2, rank=0, stream="probe:8")
        again = crop_batches(data, 4, seed=1, step=2, rank=0, stream="train")
        full = crop_batches(data, 16, seed=1, step=2, rank=0, stream="train")
        for a, b, c, d in zip(first, again, full, data, strict=True):
            torch.testing.assert_close(a[0]["input"], b[0]["input"], rtol=0, atol=0)
            torch.testing.assert_close(c[1], d[1], rtol=0, atol=0)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)


class MonitorTest(unittest.TestCase):
    def test_direct_vector_distances_missing_and_zero_gradients(self):
        p = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
        q = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        monitor = SeqGradMonitor([("p", p), ("q", q), ("alias", p)])
        monitor.CHUNK_ELEMENTS = 3
        p.grad = torch.arange(7, dtype=torch.float64)
        ref = torch.cat([p.grad.clone(), torch.zeros(2)])
        self.assertEqual(monitor.capture(), ref.norm().item())
        self.assertEqual(monitor.compare()["distance"], 0)
        p.grad = None
        q.grad = torch.tensor([2.0, 3.0], dtype=torch.float64)
        actual = torch.cat([torch.zeros(7), q.grad])
        values = monitor.compare()
        self.assertAlmostEqual(values["distance_sq"], (actual - ref).square().sum().item())
        self.assertAlmostEqual(
            values["relative_distance"], (actual - ref).norm().item() / (ref.norm().item() + 1e-12)
        )
        self.assertEqual(values["cosine"], 0)
        q.grad = None
        self.assertEqual(monitor.compare()["cosine_defined"], 0)
        q.grad = torch.full_like(q, float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            monitor.compare()


class TrainerTest(unittest.TestCase):
    def test_periodic_monitoring_preserves_updates_and_omits_skipped_distances(self):
        off, periodic = probe_trainer(False), probe_trainer()
        periodic.config.seq_grad.interval = 5
        off.config.training.steps = periodic.config.training.steps = 12
        for step in range(1, 13):
            off.step = periodic.step = step
            data = parents()
            off.train_step(off.batch_generator(data))
            periodic.train_step(periodic.batch_generator(data))
            payload = periodic.metrics_processor.logger.log.call_args.args[0]
            monitored = step in (1, 5, 10, 12)
            self.assertEqual("seq_grad_distance/s00004" in payload, monitored)
            self.assertEqual(payload["seq_grad/samples_per_length"], int(monitored))
            self.assertEqual(payload["seq_grad_cost/probe_tokens"] > 0, monitored)
            torch.testing.assert_close(
                off.model_parts[0].weight, periodic.model_parts[0].weight, rtol=0, atol=0
            )
        self.assertEqual(off.ntokens_seen, periodic.ntokens_seen)

    def test_accumulation_partition_preserves_global_batch_distances(self):
        data = parents()
        combined = [
            (
                {"input": torch.cat([inputs["input"] for inputs, _ in data])},
                torch.cat([labels for _, labels in data]),
            )
        ]
        trainers = [probe_trainer(k=2, batch=2), probe_trainer(k=1, batch=4)]
        payloads = []
        for trainer, batches in zip(trainers, [data, combined], strict=True):
            trainer.train_step(trainer.batch_generator(batches))
            payloads.append(trainer.metrics_processor.logger.log.call_args.args[0])
        for key, value in payloads[0].items():
            if key.startswith(("seq_grad_distance/", "seq_grad_reference/")):
                self.assertAlmostEqual(value, payloads[1][key], places=12)
        torch.testing.assert_close(
            trainers[0].model_parts[0].weight, trainers[1].model_parts[0].weight
        )

    def test_all_distances_match_explicit_gradients_and_token_normalization(self):
        trainer = probe_trainer()
        data = parents()
        data[0][1][0, 1] = -100
        model = copy.deepcopy(trainer.model_parts[0])
        reference = explicit_gradient(model, data)
        expected = {}
        for length in [2, 4, 8]:
            batches = trainer._crop(data, length, "train" if length == 4 else f"probe:{length}")
            grad = explicit_gradient(model, batches)
            expected[length] = (grad - reference).norm().item()
        trainer.train_step(trainer.batch_generator(data))
        payload, step = trainer.metrics_processor.logger.log.call_args.args
        self.assertEqual(step, 1)
        for length, distance in expected.items():
            self.assertAlmostEqual(payload[f"seq_grad_distance/s{length:05d}"], distance, places=12)
        self.assertEqual(payload["seq_grad_reference/valid_tokens"], 63)
        self.assertEqual(payload["n_tokens_seen"], 16)
        self.assertEqual(payload["seq_grad_cost/probe_tokens"], 104)
        self.assertEqual(payload["seq_grad_cost/forward_backward_tokens"], 120)
        self.assertEqual(len(trainer.optimizers.gradients), 1)
        self.assertIsNone(trainer.metrics_processor.pending)
        self.assertFalse(trainer.seq_grad_monitor.reference)

    def test_probes_preserve_updates_optimizer_rng_buffers_and_lr(self):
        off, on = probe_trainer(False, RandomModel()), probe_trainer(True, RandomModel())
        for step in range(1, 4):
            off.step = on.step = step
            state = torch.get_rng_state()
            off.train_step(off.batch_generator(parents()))
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(state)
            on.train_step(on.batch_generator(parents()))
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            for key, value in off.model_parts[0].state_dict().items():
                torch.testing.assert_close(
                    value, on.model_parts[0].state_dict()[key], rtol=0, atol=0
                )
            for a, b in zip(
                off.optimizers.optimizer.state.values(),
                on.optimizers.optimizer.state.values(),
                strict=True,
            ):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertEqual(
                off.lr_schedulers.scheduler.state_dict(), on.lr_schedulers.scheduler.state_dict()
            )
            self.assertEqual(off.ntokens_seen, on.ntokens_seen)
            self.assertEqual(off.n_valid_tokens_seen, on.n_valid_tokens_seen)

    def test_failure_prevents_update_and_clears_pending_metrics(self):
        for failure in ("exhaustion", "shape", "nan", "no_tokens", "input_shape"):
            with self.subTest(failure=failure):
                trainer = probe_trainer()
                data = parents()
                if failure == "exhaustion":
                    data.pop()
                elif failure == "shape":
                    data[0] = (data[0][0], data[0][1][:, :3])
                elif failure == "nan":
                    data[0][0]["input"].fill_(float("nan"))
                elif failure == "input_shape":
                    data[0][0]["input"] = torch.zeros(1, 16)
                else:
                    for _, labels in data:
                        labels.fill_(-100)
                with self.assertRaises((ValueError, RuntimeError, DataloaderExhaustedError)):
                    trainer.train_step(trainer.batch_generator(data))
                self.assertEqual(len(trainer.optimizers.gradients), 0)
                self.assertEqual(trainer.ntokens_seen, 0)
                self.assertEqual(trainer.n_valid_tokens_seen, 0)
                self.assertIsNone(trainer.metrics_processor.pending)
                trainer.train_step(trainer.batch_generator(parents()))
                self.assertEqual(len(trainer.optimizers.gradients), 1)

    def test_resume_preserves_crops_and_update(self):
        full = probe_trainer()
        full.train_step(full.batch_generator(parents()))
        resumed = probe_trainer()
        resumed.load_state_dict(copy.deepcopy(full.state_dict()))
        resumed.model_parts[0].load_state_dict(full.model_parts[0].state_dict())
        resumed.optimizers.optimizer.load_state_dict(
            copy.deepcopy(full.optimizers.optimizer.state_dict())
        )
        resumed.lr_schedulers.scheduler.load_state_dict(full.lr_schedulers.scheduler.state_dict())
        for trainer in (full, resumed):
            trainer.step = 2
            trainer.train_step(trainer.batch_generator(parents()))
        torch.testing.assert_close(
            full.model_parts[0].weight, resumed.model_parts[0].weight, rtol=0, atol=0
        )
        self.assertEqual(full.state_dict(), resumed.state_dict())
        resumed.config.seq_grad.seed += 1
        with self.assertRaisesRegex(ValueError, "protocol"):
            resumed.load_state_dict(full.state_dict())


class LoggingAndConfigTest(unittest.TestCase):
    def test_minimal_wandb_keeps_distances_train_validation_loss_and_time(self):
        wrapped = Mock()
        logger = SeqGradWandBLogger(wrapped, profile_freq=1, minimal=True)
        metrics = {
            "seq_grad_distance/s01024": 2.0,
            "seq_grad_distance_sq/s01024": 4.0,
            "seq_grad_relative_distance/s01024": 0.5,
            "loss_metrics/global_avg_loss": 3.0,
            "time_metrics/end_to_end(s)": 1.2,
            "memory/max_reserved(GiB)": 100.0,
        }
        original = metrics.copy()
        logger.log(metrics, 5)
        wrapped.log.assert_called_once_with(
            {
                "context-truncation bound/seq_1024": 2.0,
                "context-truncation bound sq/seq_1024": 4.0,
                "train/loss": 3.0,
                "time/step_seconds": 1.2,
            },
            5,
        )
        self.assertEqual(metrics, original)
        wrapped.wandb.plot.line_series.assert_not_called()
        logger.log({"validation_metrics/loss": 2.9}, 5)
        wrapped.log.assert_called_with({"val/loss": 2.9}, 5)
        logger.log({"memory/max_reserved(GiB)": 100}, 6)
        self.assertEqual(wrapped.log.call_count, 2)

    def test_wandb_profiles_share_scalar_commit_and_skip_validation(self):
        wrapped = Mock()
        logger = SeqGradWandBLogger(wrapped, profile_freq=2)
        payload = {
            f"seq_grad_{name}/s{length:05d}": float(length)
            for name in ("distance", "relative_distance", "cosine")
            for length in [8, 2, 4]
        }
        logger.log(payload, 1)
        self.assertEqual(wrapped.log.call_count, 1)
        self.assertEqual(wrapped.wandb.plot.line_series.call_count, 3)
        self.assertEqual(wrapped.wandb.plot.line_series.call_args.kwargs["xs"], [2, 4, 8])
        self.assertNotIn("seq_grad_profile/distance", payload)
        logger.log({"validation/loss": 2}, 2)
        logger.log(payload, 3)
        self.assertEqual(wrapped.wandb.plot.line_series.call_count, 3)

    def test_launch_configs_and_validation(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("smoke", "reference16384"):
            raw, _, _, args = resolve_point(root / f"configs/seq_grad/{name}.yaml", 0)
            with patch.dict(
                "os.environ",
                {"DYNAMIC_SEQ_LEN_VALIDATION_DIR": raw["paths"].get("validation_dir", "")},
            ):
                cfg = ConfigManager().parse_args(
                    [
                        "--module",
                        "torchtitan_dynamic_seq_len.config_registry",
                        "--config",
                        raw["torchtitan_config"],
                        *args,
                    ]
                )
            self.assertIs(cfg._owner, SeqGradTrainer)
            cfg.validate_seq_grad()
            if name == "reference16384":
                self.assertEqual(cfg.training.global_batch_size, 128)
                self.assertEqual(cfg.training.seq_len, 16384)
                self.assertEqual(cfg.seq_grad.train_length, 2048)
                self.assertEqual(len(cfg.seq_grad.lengths), 11)
                self.assertEqual(sum([16384, *cfg.seq_grad.lengths]) / 2048, 17)
            cfg.metrics.log_freq = 2
            with self.assertRaisesRegex(ValueError, "log_freq"):
                cfg.validate_seq_grad()


if __name__ == "__main__":
    unittest.main()
