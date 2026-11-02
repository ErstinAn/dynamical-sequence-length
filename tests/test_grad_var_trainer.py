"""Exercise the actual TorchTitan train_step/backward/logging on a tiny CPU model.

Only GPU-dependent initialization is replaced by a fixture. The training loop,
loss normalization, clipping and variance integration execute production methods.
"""

import copy
import time
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.distributed.parallel_dims import ParallelDims

from torchtitan_dynamic_seq_len.grad_var.metrics import GradVarMetricsProcessor
from torchtitan_dynamic_seq_len.grad_var.monitor import GradVarMonitorConfig
from torchtitan_dynamic_seq_len.grad_var.trainer import GradVarTrainer


def cpu_metrics(dims, log_freq=1):
    metrics = object.__new__(GradVarMetricsProcessor)
    metrics.config = GradVarMetricsProcessor.Config(log_freq=log_freq)
    metrics.parallel_dims = dims
    metrics.clear_grad_var()
    metrics.logger = Mock()
    metrics.num_flops_per_token = 1
    metrics.gpu_peak_flops = 1e12
    metrics.has_quantization = False
    metrics.time_last_log = time.perf_counter()
    metrics.ntokens_since_last_log = 1
    metrics.data_loading_times = [0.001]
    metrics.device_memory_monitor = Mock()
    metrics.device_memory_monitor.get_peak_stats.return_value = SimpleNamespace(
        max_active_gib=0,
        max_active_pct=0,
        max_reserved_gib=0,
        max_reserved_pct=0,
        num_alloc_retries=0,
        num_ooms=0,
    )
    metrics.color = SimpleNamespace(
        **dict.fromkeys(
            ("red", "green", "orange", "turquoise", "blue", "cyan", "magenta", "reset", "yellow"),
            "",
        )
    )
    return metrics


class OptimizerFixture:
    def __init__(self, model):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        self.model = model
        self.gradients = []

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.gradients.append([p.grad.detach().clone() for p in self.model.parameters()])
        self.optimizer.step()


def tiny_trainer(enabled=True, *, k=4, b=2, seq=3, log_freq=1, model=None):
    trainer = object.__new__(GradVarTrainer)
    trainer.config = SimpleNamespace(
        grad_var_monitor=GradVarMonitorConfig(enable=enabled),
        training=SimpleNamespace(
            local_batch_size=b, global_batch_size=b * k, seq_len=seq, max_norm=0.01
        ),
        parallelism=SimpleNamespace(spmd_backend="default"),
    )
    trainer.parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, tp=1, cp=1, pp=1, ep=1, world_size=1
    )
    trainer.parallel_dims.get_optional_mesh = lambda name: None
    trainer.model_parts = [model if model is not None else torch.nn.Linear(seq, seq).double()]
    trainer.model_config = None
    trainer.device = torch.device("cpu")
    trainer.train_context = nullcontext
    trainer.gradient_accumulation_steps = k
    trainer.num_pipeline_parallel_microbatches = 1
    trainer.step = 1
    trainer.ntokens_seen = 0
    trainer.metrics_processor = cpu_metrics(trainer.parallel_dims, log_freq)
    trainer.optimizers = OptimizerFixture(trainer.model_parts[0])
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizers.optimizer, lambda n: 0.95**n)
    trainer.lr_schedulers = SimpleNamespace(
        step=scheduler.step,
        get_metrics=lambda: {"lr": scheduler.get_last_lr()[0]},
        scheduler=scheduler,
    )
    trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)

    def loss_fn(pred, labels, global_valid_tokens):
        mask = labels != IGNORE_INDEX
        return ((pred - labels)[mask].square().sum() / global_valid_tokens, None)

    trainer.loss_fn = loss_fn
    trainer._init_grad_var_monitor()
    return trainer


def batches(k=4, b=2, seq=3):
    rng = torch.Generator().manual_seed(14)
    return [
        (
            {"input": torch.randn(b, seq, generator=rng, dtype=torch.float64)},
            torch.randn(b, seq, generator=rng, dtype=torch.float64),
        )
        for _ in range(k)
    ]


def explicit_reference(model, data):
    """Fresh forwards at fixed theta, each normalized by its own group tokens."""
    samples = [
        torch.autograd.grad(
            (model(inputs["input"]) - labels).square().mean(), tuple(model.parameters())
        )
        for inputs, labels in data
    ]
    result = {}
    for i, (name, _) in enumerate(model.named_parameters()):
        stacked = torch.stack([sample[i] for sample in samples])
        result[f"grad_var_monitor/grad_var_monitor_{name}_var"] = (
            (stacked - stacked.mean(dim=0)).square().sum() / (len(data) - 1)
        ).item()
    result["global_grad_var"] = sum(result.values())
    return result


class GradVarTrainerTest(unittest.TestCase):
    def test_preclip_reference_and_exact_training_equivalence(self):
        for k, b, seq in ((2, 3, 4), (4, 2, 3), (8, 1, 2)):
            with self.subTest(k=k):
                off = tiny_trainer(False, k=k, b=b, seq=seq)
                on = tiny_trainer(True, k=k, b=b, seq=seq, model=copy.deepcopy(off.model_parts[0]))
                data = batches(k=k, b=b, seq=seq)
                for step in (1, 2):
                    reference = explicit_reference(on.model_parts[0], data)
                    rng = torch.get_rng_state().clone()
                    for trainer in (off, on):
                        trainer.step = step
                        trainer.metrics_processor.data_loading_times = [0.001]
                        iterator = iter(data)
                        trainer.train_step(iterator)
                        self.assertIsNone(next(iterator, None))
                    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                    self.assertEqual(off.ntokens_seen, on.ntokens_seen)
                    for p, q in zip(
                        off.model_parts[0].parameters(), on.model_parts[0].parameters(), strict=True
                    ):
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                    for left, right in zip(
                        off.optimizers.optimizer.state.values(),
                        on.optimizers.optimizer.state.values(),
                        strict=True,
                    ):
                        for key in left:
                            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                    self.assertEqual(
                        off.lr_schedulers.scheduler.state_dict(),
                        on.lr_schedulers.scheduler.state_dict(),
                    )
                    payload, logged_step = on.metrics_processor.logger.log.call_args.args
                    self.assertEqual(logged_step, step)
                    for key, value in reference.items():
                        self.assertAlmostEqual(payload[key], value, places=10)
                    self.assertGreater(payload["global_grad_var"], 0.01**2)
                    self.assertIsNone(on.metrics_processor._grad_var_pending)
                    self.assertEqual(on.grad_var_monitor.count, 0)
                self.assertEqual(on.metrics_processor.logger.log.call_count, 2)
                self.assertNotIn(
                    "global_grad_var", off.metrics_processor.logger.log.call_args.args[0]
                )

    def test_cadence_recovery_and_validation_do_not_leak(self):
        trainer = tiny_trainer(log_freq=3)
        data = batches()
        trainer.train_step(iter(data))
        snapshot = trainer.grad_var_monitor.previous["weight"].clone()
        trainer.step = 2
        with patch.object(
            trainer.grad_var_monitor, "observe", wraps=trainer.grad_var_monitor.observe
        ) as observe:
            trainer.train_step(iter(data))
            observe.assert_not_called()
        torch.testing.assert_close(
            snapshot, trainer.grad_var_monitor.previous["weight"], rtol=0, atol=0
        )
        self.assertEqual(trainer.metrics_processor.logger.log.call_count, 1)
        trainer.metrics_processor.log_validation(0.4, 2)
        self.assertNotIn("global_grad_var", trainer.metrics_processor.logger.log.call_args.args[0])
        self.assertEqual(set(trainer.state_dict()), {"step", "ntokens_seen"})
        trainer.load_state_dict({"step": 5, "ntokens_seen": 900})
        trainer.step += 1
        trainer.metrics_processor.data_loading_times = [0.001]
        trainer.train_step(iter(data))
        self.assertEqual(trainer.metrics_processor.logger.log.call_args.args[1], 6)
        self.assertIn("global_grad_var", trainer.metrics_processor.logger.log.call_args.args[0])

    def test_errors_before_update_and_next_step_reset(self):
        for error_kind in ("tokens", "shape", "nan", "exhaustion"):
            with self.subTest(error_kind=error_kind):
                trainer = tiny_trainer()
                data = batches()
                initial = copy.deepcopy(trainer.model_parts[0].state_dict())
                if error_kind == "tokens":
                    data[0][1][0, 0] = IGNORE_INDEX
                elif error_kind == "shape":
                    data[0] = ({"input": data[0][0]["input"][:1]}, data[0][1][:1])
                elif error_kind == "nan":
                    data[0][0]["input"][0, 0] = float("nan")
                else:
                    data = data[:1]
                with self.assertRaises((RuntimeError, StopIteration)):
                    trainer.train_step(iter(data))
                self.assertEqual(len(trainer.optimizers.gradients), 0)
                self.assertEqual(trainer.metrics_processor.logger.log.call_count, 0)
                self.assertIsNone(trainer.metrics_processor._grad_var_pending)
                self.assertFalse(trainer.grad_var_monitor.active)
                for key, value in initial.items():
                    torch.testing.assert_close(
                        value, trainer.model_parts[0].state_dict()[key], rtol=0, atol=0
                    )
                trainer.train_step(iter(batches()))
                self.assertEqual(trainer.metrics_processor.logger.log.call_count, 1)

    def test_config_owner_both_registry_templates_and_build(self):
        from torchtitan_dynamic_seq_len.config_registry import (
            llama3_100m_llama2_adamw,
            llama3_100m_llama2_muon,
        )

        for factory in (llama3_100m_llama2_adamw, llama3_100m_llama2_muon):
            config = factory()
            self.assertIs(config._owner, GradVarTrainer)
            self.assertIs(config.metrics._owner, GradVarMetricsProcessor)
            self.assertFalse(config.grad_var_monitor.enable)
            with patch.object(GradVarTrainer, "__init__", return_value=None):
                self.assertIsInstance(config.build(), GradVarTrainer)
        self.assertIs(MetricsProcessor.Config._owner, MetricsProcessor)

    def test_disabled_k1_and_enabled_unsupported_runtime(self):
        trainer = tiny_trainer(False, k=1)
        trainer.train_step(iter(batches(k=1)))
        self.assertIsNone(trainer.grad_var_monitor)
        with self.assertRaisesRegex(ValueError, ">= 2"):
            tiny_trainer(True, k=1)
        trainer = tiny_trainer(False)
        trainer.config.grad_var_monitor.enable = True
        trainer.parallel_dims.dp_shard = 2
        with self.assertRaisesRegex(ValueError, "replicated DP"):
            trainer._init_grad_var_monitor()

    def test_logging_merges_once_and_checks_step(self):
        metrics = cpu_metrics(tiny_trainer(False).parallel_dims)
        metrics.queue_grad_var(1, {"global_grad_var": 4.0})
        metrics.log(1, 1.0, 1.0, 1.0, {"lr": 0.1})
        self.assertEqual(metrics.logger.log.call_count, 1)
        self.assertEqual(metrics.logger.log.call_args.args[0]["global_grad_var"], 4.0)
        metrics.queue_grad_var(2, {"global_grad_var": 5.0})
        with self.assertRaisesRegex(RuntimeError, "different step"):
            metrics.log(3, 1.0, 1.0, 1.0)
        self.assertIsNone(metrics._grad_var_pending)


if __name__ == "__main__":
    unittest.main()
