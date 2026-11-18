"""Independent sampling, actual loss normalization, state and event contracts."""

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.config import ConfigManager

from tests.test_document_packing import fixture
from tests.test_dynamic_seq import ScalarModel
from tests.test_grad_var_trainer import batches, cpu_metrics, tiny_trainer
from torchtitan_dynamic_seq_len.data.packed_documents import PackedDocumentDataset
from torchtitan_dynamic_seq_len.run_config import resolve_point
from torchtitan_dynamic_seq_len.seq_grad.metrics import SeqGradMetricsProcessor, SeqGradWandBLogger
from torchtitan_dynamic_seq_len.seq_grad.monitor import LENGTHS, SeqGradConfig, SeqGradMonitor
from torchtitan_dynamic_seq_len.seq_grad.sampler import PackedProbeLoader, stream_seed
from torchtitan_dynamic_seq_len.seq_grad.trainer import SeqGradTrainer


class SyntheticSampler:
    identity = {"protocol": "synthetic-independent-test"}
    def sample_batch(self, *, length, global_sequences, local_batch_size, rank, degree, seed, step):
        rows = []
        for i in range(rank, global_sequences, degree):
            rng = torch.Generator().manual_seed(stream_seed(seed, step, length, i))
            x = torch.randn(length, dtype=torch.float64, generator=rng)
            y = torch.randn(length, dtype=torch.float64, generator=rng)
            y[:i % length] = -100  # Unequal valid counts across ranks/microbatches.
            rows.append((x, y))
        return [({"input": torch.stack([x for x, _ in rows[j:j+local_batch_size]])},
                 torch.stack([y for _, y in rows[j:j+local_batch_size]]))
                for j in range(0, len(rows), local_batch_size)]


def probe_trainer(enabled=True, model=None):
    base = tiny_trainer(False, k=2, b=2, seq=4, model=model or ScalarModel(), log_freq=3)
    trainer = object.__new__(SeqGradTrainer)
    trainer.__dict__.update(base.__dict__)
    trainer.config.seq_grad = SeqGradConfig(enable=enabled, interval=2, reference_length=16,
                                          lengths=[2, 4, 8], probe_global_batch_size=8,
                                          probe_local_batch_size=2)
    trainer.config.training.steps = 4
    trainer._probe_active, trainer._last_event, trainer._signature = False, -1, "fixture"
    trainer._training_calls = trainer.n_valid_tokens_seen = 0
    trainer._probe_rank, trainer._probe_degree = 0, 1
    trainer.probe_loader = SyntheticSampler()
    trainer.seq_grad_monitor = SeqGradMonitor(trainer.model_parts[0].named_parameters())
    metrics = object.__new__(SeqGradMetricsProcessor)
    metrics.__dict__.update(cpu_metrics(trainer.parallel_dims, 3).__dict__)
    trainer.metrics_processor = metrics
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizers.optimizer, lambda n: 1.0)
    trainer.lr_schedulers.scheduler = scheduler
    trainer.lr_schedulers.step = scheduler.step
    trainer.lr_schedulers.get_metrics = lambda: {"lr": scheduler.get_last_lr()[0]}
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
        random.random()
        np.random.random()
        return super().forward(inputs, **kwargs) * torch.rand_like(inputs)


class SamplerTests(unittest.TestCase):
    def test_streams_packing_dp_partition_and_training_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            source = PackedDocumentDataset(root, split="train", seq_len=16, infinite=True)
            iterator = iter(source)
            next(iterator)
            state = copy.deepcopy(source.state_dict())
            sampler = PackedProbeLoader(source)
            kwargs = dict(length=16, global_sequences=8, local_batch_size=2,
                          rank=0, degree=1, seed=3, step=2)
            first = sampler.sample_batch(**kwargs)
            again = sampler.sample_batch(**kwargs)
            short = sampler.sample_batch(**{**kwargs, "length": 8})
            self.assertEqual(source.state_dict(), state)
            for a, b in zip(first, again):
                torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
                self.assertTrue((a[0]["positions"][:, 0] == 0).all())
                self.assertTrue((a[1][a[1] != -100] != source.bos).all())
            self.assertTrue(any(not torch.equal(a[0]["input"][:, :8], b[0]["input"])
                                for a, b in zip(first, short)))
            whole = torch.cat([labels for _, labels in first])
            for rank in range(2):
                part = sampler.sample_batch(**{**kwargs, "rank": rank, "degree": 2})
                torch.testing.assert_close(torch.cat([y for _, y in part]), whole[rank::2])
            restored = PackedDocumentDataset(root, split="train", seq_len=16, infinite=True)
            restored.load_state_dict(state)
            expected, actual = next(iter(restored)), next(iterator)
            torch.testing.assert_close(expected[1], actual[1])
            # Production iterator is the same packing path at a selected start.
            token_seed = stream_seed(3, 2, 16, 0)
            direct = sampler._sample(16, token_seed)
            torch.testing.assert_close(direct[0]["positions"], first[0][0]["positions"][0])
            self.assertTrue(any((y == -100).any() for _, y in first))


class MonitorTests(unittest.TestCase):
    def test_chunked_missing_tied_and_nonfinite(self):
        p = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
        q = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        monitor = SeqGradMonitor([("p", p), ("q", q), ("alias", p)])
        monitor.CHUNK_ELEMENTS = 3
        p.grad = torch.arange(7, dtype=torch.float64)
        ref = torch.cat([p.grad.clone(), torch.zeros(2)])
        monitor.capture()
        self.assertEqual(monitor.distance(), 0)
        p.grad, q.grad = None, torch.tensor([2., 3.], dtype=torch.float64)
        self.assertAlmostEqual(monitor.distance(), (torch.cat([torch.zeros(7), q.grad])-ref).norm().item())
        q.grad.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            monitor.distance()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            monitor.capture()

    def test_config_guards_and_grid(self):
        self.assertEqual(len(LENGTHS), 16)
        for dp in (1, 2, 4):
            SeqGradConfig().validate(dp)
        for values in ({"interval": 0}, {"lengths": [1, 1]}, {"lengths": [20000]},
                       {"probe_global_batch_size": 7}, {"probe_local_batch_size": 0}):
            with self.assertRaises(ValueError):
                SeqGradConfig(**values).validate(4)


class TrainerTests(unittest.TestCase):
    def test_full_grid_and_one_reference_per_event(self):
        trainer = probe_trainer()
        trainer.config.seq_grad.lengths = list(LENGTHS)
        trainer.config.seq_grad.reference_length = 16384
        sampler = trainer.probe_loader.sample_batch
        trainer.probe_loader.sample_batch = Mock(side_effect=sampler)
        trainer.maybe_monitor(0)
        payload = trainer.metrics_processor.logger.log.call_args.args[0]
        self.assertEqual(set(payload), {"context_bias/step", *[f"context_bias/distance_s{s}" for s in LENGTHS]})
        calls = trainer.probe_loader.sample_batch.call_args_list
        self.assertEqual([call.kwargs["length"] for call in calls], [16384, *LENGTHS])

    def test_native_constant_scheduler_preserves_group_multipliers(self):
        p, q = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([{"params": [p], "lr": 0.0005},
                                       {"params": [q], "lr": 0.001}])
        scheduler = LRSchedulersContainer.Config(warmup_steps=0, min_lr_factor=1.0).build(
            optimizers=[optimizer], training_steps=3819)
        for step in range(3820):
            self.assertEqual([g["lr"] for g in optimizer.param_groups], [0.0005, 0.001])
            optimizer.step()
            scheduler.step()

    def test_raw_distances_and_accumulation(self):
        reference = probe_trainer()
        cfg = reference.config.seq_grad
        direct = {}
        for length in [cfg.reference_length, *cfg.lengths]:
            data = reference.probe_loader.sample_batch(length=length, global_sequences=8,
                local_batch_size=8, rank=0, degree=1, seed=0, step=0)
            direct[length] = explicit_gradient(copy.deepcopy(reference.model_parts[0]), data)
        for size in (1, 2, 4, 8):
            trainer = probe_trainer()
            trainer.config.seq_grad.probe_local_batch_size = size
            trainer.maybe_monitor(0)
            payload = trainer.metrics_processor.logger.log.call_args.args[0]
            self.assertEqual(set(payload), {"context_bias/step", *[f"context_bias/distance_s{s}" for s in cfg.lengths]})
            for length in cfg.lengths:
                self.assertAlmostEqual(payload[f"context_bias/distance_s{length}"],
                    (direct[length]-direct[16]).norm().item(), places=12)
            self.assertFalse(trainer.seq_grad_monitor.reference)
            self.assertEqual(trainer.ntokens_seen, 0)

    def test_state_rng_optimizer_lr_events_and_resume(self):
        off, on = probe_trainer(False, RandomModel()), probe_trainer(True, RandomModel())
        for step in range(1, 5):
            torch_state, py_state, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
            off.step = on.step = step
            off.train_step(off.batch_generator(batches(k=2, b=2, seq=4)))
            after = torch.get_rng_state(), random.getstate(), np.random.get_state()
            torch.set_rng_state(torch_state)
            random.setstate(py_state)
            np.random.set_state(np_state)
            on.train_step(on.batch_generator(batches(k=2, b=2, seq=4)))
            torch.testing.assert_close(torch.get_rng_state(), after[0], rtol=0, atol=0)
            self.assertEqual(random.getstate(), after[1])
            np.testing.assert_equal(np.random.get_state(), after[2])
            for name, value in off.model_parts[0].state_dict().items():
                torch.testing.assert_close(value, on.model_parts[0].state_dict()[name], rtol=0, atol=0)
            for a, b in zip(off.optimizers.optimizer.state.values(), on.optimizers.optimizer.state.values()):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertEqual(on.lr_schedulers.get_metrics()["lr"], 0.03)
        events = [call.args[0] for call in on.metrics_processor.logger.log.call_args_list
                  if "context_bias/step" in call.args[0]]
        self.assertEqual([p["context_bias/step"] for p in events], [0, 2, 4])
        self.assertEqual(off.ntokens_seen, on.ntokens_seen)
        self.assertEqual(off.n_valid_tokens_seen, on.n_valid_tokens_seen)
        state = copy.deepcopy(on.state_dict())
        resumed = probe_trainer()
        resumed.load_state_dict(state)
        resumed.maybe_monitor(4)
        resumed.metrics_processor.logger.log.assert_not_called()

    def test_error_restores_buffers_rng_and_reference(self):
        trainer = probe_trainer(model=RandomModel())
        saved = torch.get_rng_state().clone()
        state = copy.deepcopy(trainer.model_parts[0].state_dict())
        original = trainer._statistics
        def fail(capture=False):
            if not capture:
                raise RuntimeError("test failure")
            return original(capture=True)
        trainer._statistics = fail
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            trainer.maybe_monitor(0)
        torch.testing.assert_close(saved, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in state.items():
            torch.testing.assert_close(value, trainer.model_parts[0].state_dict()[key], rtol=0, atol=0)
        self.assertFalse(trainer._probe_active)
        self.assertFalse(trainer.seq_grad_monitor.reference)
        self.assertEqual(trainer._last_event, -1)
        self.assertTrue(all(p.grad is None for p in trainer.model_parts[0].parameters()))

    def test_logger_preserves_names_and_open_step(self):
        wrapped = Mock(tag=None)
        logger = SeqGradWandBLogger(wrapped, {"seed": 0})
        payload = {"context_bias/step": 0, "context_bias/distance_s64": 1.2}
        logger.log(payload, 0)
        wrapped.wandb.log.assert_called_with(payload, step=0, commit=False)
        normal = {"loss_metrics/global_avg_loss": 1.}
        logger.log(normal, 2)
        wrapped.wandb.log.assert_called_with(normal, step=2, commit=False)
        logger.close()
        wrapped.wandb.run.define_metric.assert_any_call("context_bias/distance_s*", step_metric="context_bias/step")

    def test_example_configs(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("smoke", "reference16384", "variance_monitor", "variance_monitor_smoke",
                     "variance_monitor_train16384", "variance_monitor_train16384_smoke",
                     "variance_monitor_paired_train16384", "variance_monitor_paired_train16384_smoke",
                     "variance_monitor_paired_document_train16384", "variance_monitor_paired_document_train16384_smoke",
                     "variance_monitor_shuffled_v1_train16384_val22256", "variance_monitor_shuffled_v1_train16384_val22256_smoke"):
            raw, _, _, args = resolve_point(root / f"legacy/configs/seq_grad/{name}.yaml", 0)
            cfg = ConfigManager().parse_args(["--module", "torchtitan_dynamic_seq_len.config_registry",
                "--config", raw["torchtitan_config"], *args])
            cfg.validate_seq_grad()


if __name__ == "__main__":
    unittest.main()
