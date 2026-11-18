"""Verify two-phase loss, independent batches, schedules, gradients and restoration."""

import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import ConfigManager
from torchtitan.trainer import Trainer

from tests.test_dynamic_seq import ScalarModel, dynamic_trainer
from tests.test_dynamic_seq_bs import batch_loader
from tests.test_grad_var_trainer import OptimizerFixture, tiny_trainer
from torchtitan_dynamic_seq_len.config_registry import llama3_debug_loss_grad_seq
from torchtitan_dynamic_seq_len.loss_grad_seq.batching import plan_batches, regroup_batches
from torchtitan_dynamic_seq_len.loss_grad_seq.metrics import LossGradMetricsProcessor
from torchtitan_dynamic_seq_len.loss_grad_seq.trainer import LossGradSeqTrainer
from torchtitan_dynamic_seq_len.run_config import resolve_point
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig


def dual_trainer(log_freq=1, dynamic_batch=False):
    trainer = dynamic_trainer(log_freq=log_freq)
    trainer.__class__ = LossGradSeqTrainer
    with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
        config = llama3_debug_loss_grad_seq()
    config.training.seq_len = 8
    config.training.steps = 12
    config.training.max_norm = 1e12
    config.loss.num_chunks = 1
    config.loss_seq_len_scheduler = SeqLenSchedulerConfig(length=8)
    config.grad_seq_len_scheduler = SeqLenSchedulerConfig(
        kind="cyclic", min_length=2, max_length=8, period_steps=12
    )
    if dynamic_batch:
        config.batch_size_scheduler = BatchSizeSchedulerConfig(
            kind="increasing", batch_sizes=[8, 16], boundaries=[6]
        )
        config.loss_seq_len_scheduler = SeqLenSchedulerConfig(
            kind="cyclic", min_length=6, max_length=8, period_steps=4
        )
        config.grad_seq_len_scheduler.max_length = 4
        config.grad_seq_len_scheduler.period_steps = 4
        config.loss_microbatch_scheduler = BatchSizeSchedulerConfig(
            kind="increasing", batch_sizes=[1, 2], boundaries=[6]
        )
        config.grad_microbatch_scheduler = BatchSizeSchedulerConfig(
            kind="increasing", batch_sizes=[4, 8], boundaries=[6]
        )
    trainer.config = config
    trainer.schedules = config.validate_loss_grad()
    trainer.parent_tokens_seen = trainer.parent_valid_tokens_seen = 0
    trainer.flops_by_length = dict.fromkeys(range(2, 9), 30)
    trainer.metrics_processor.__class__ = LossGradMetricsProcessor
    trainer.dataloader = batch_loader(infinite=True)
    return trainer


class IndependentBatchTest(unittest.TestCase):
    def test_phase_capacities_are_independent(self):
        p = plan_batches(
            loss_length=12288,
            grad_length=512,
            global_batch=512,
            dp=2,
            loss_limit=256,
            grad_limit=256,
            loss_token_budget=98304,
            grad_token_budget=32768,
        )
        self.assertEqual((p.loss_microbatch, p.grad_microbatch), (8, 64))
        self.assertEqual((p.loss_batches, p.grad_batches), (32, 4))
        p = plan_batches(
            loss_length=12288,
            grad_length=12288,
            global_batch=512,
            dp=2,
            loss_limit=256,
            grad_limit=256,
            loss_token_budget=98304,
            grad_token_budget=32768,
        )
        self.assertEqual((p.loss_microbatch, p.grad_microbatch), (8, 2))
        with self.assertRaisesRegex(ValueError, "one sequence"):
            plan_batches(
                loss_length=8,
                grad_length=4,
                global_batch=8,
                dp=1,
                loss_limit=8,
                grad_limit=8,
                loss_token_budget=4,
            )

    def test_invalid_intermediate_lengths_and_chunk_alignment(self):
        trainer = dual_trainer()
        trainer.config.grad_seq_len_scheduler.max_length = 14
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            trainer.config.validate_loss_grad()
        trainer = dual_trainer()
        trainer.config.loss.num_chunks = 2
        with self.assertRaisesRegex(ValueError, "loss chunks"):
            trainer.config.validate_loss_grad()

    def test_crop_independent_of_loss_partition_and_preserves_alignment(self):
        trainer = dual_trainer()
        trainer._prepare_step()
        data = []
        for i in range(8):
            x = torch.arange(i * 8, (i + 1) * 8).reshape(1, 8)
            data.append(({"input": x, "positions": torch.arange(8).reshape(1, 8)}, x + 1))
        a = regroup_batches(trainer._crop(data), 4)
        b = regroup_batches(trainer._crop(regroup_batches(data, 4)), 4)
        for (left, ly), (right, ry) in zip(a, b):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            torch.testing.assert_close(ly, ry)
            torch.testing.assert_close(ly, left["input"] + 1)
            self.assertTrue(left["positions"][:, 0].eq(0).all())


class TwoPhaseTrainerTest(unittest.TestCase):
    def test_nondivisible_microbatches_keep_full_update_and_weighted_gradient(self):
        trainer = dual_trainer()
        trainer.config.batch_size_scheduler.batch_size = 10
        trainer.config.loss_microbatch_scheduler.batch_size = 3
        trainer.config.grad_microbatch_scheduler.batch_size = 4
        trainer.schedules = trainer.config.schedules()
        plan = trainer._prepare_step()
        parents = trainer._stage_parents(iter(trainer.dataloader))
        self.assertEqual([y.shape[0] for _, y in parents], [3, 3, 3, 1])
        crops = regroup_batches(trainer._crop(parents), plan.grad_microbatch)
        self.assertEqual([y.shape[0] for _, y in crops], [4, 4, 2])
        weight = trainer.model_parts[0].weight.detach()
        count = sum(y.numel() for _, y in crops)
        expected = sum((2 * x["input"] * (weight * x["input"] - y)).sum() for x, y in crops) / count
        trainer.train_step(trainer.batch_generator(copy.deepcopy(parents)))
        torch.testing.assert_close(
            trainer.optimizers.gradients[-1][0], expected, rtol=1e-12, atol=1e-8
        )
        self.assertEqual(trainer.ntokens_seen, count)
        self.assertEqual(trainer.parent_tokens_seen, 80)

    def test_chunked_loss_matches_full_cross_entropy_with_tied_weights(self):
        class LanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(11, 5)
                self.lm_head = torch.nn.Linear(5, 11, bias=False)
                self.lm_head.weight = self.embedding.weight

            def forward(self, inputs, **kwargs):
                return self.embedding(inputs).tanh()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            trainer = dual_trainer()
            trainer.step = 5  # crop length 6, parent length 8
            model = LanguageModel()
            trainer.model_parts = [model]
            trainer.optimizers = OptimizerFixture(model)
            trainer.loss_fn = ChunkedLossWrapper(
                ChunkedLossWrapper.Config(num_chunks=2, loss_fn=CrossEntropyLoss.Config())
            )
            trainer.loss_fn.set_lm_head(model.lm_head)
            trainer._prepare_step()
            data = [
                ({"input": torch.randint(0, 11, (1, 8))}, torch.randint(0, 11, (1, 8)))
                for _ in range(8)
            ]
            data[0][1][0, -1] = -100

            def full_ce(rows):
                total = sum(int((y != -100).sum()) for _, y in rows)
                return (
                    sum(
                        torch.nn.functional.cross_entropy(
                            model.lm_head(model(x["input"])).flatten(0, 1),
                            y.flatten(),
                            reduction="sum",
                        )
                        for x, y in rows
                    )
                    / total
                )

            expected_long = full_ce(data).item()
            crops = regroup_batches(trainer._crop(data), 4)
            expected_crop = full_ce(crops)
            gradient = torch.autograd.grad(expected_crop, model.embedding.weight)[0]
            trainer.train_step(trainer.batch_generator(data))
            torch.testing.assert_close(
                trainer.optimizers.gradients[-1][0], gradient, rtol=2e-6, atol=2e-7
            )
            metrics = trainer.metrics_processor.logger.log.call_args.args[0]
            self.assertAlmostEqual(metrics["loss_metrics/long_avg_loss"], expected_long, places=5)
            self.assertAlmostEqual(
                metrics["loss_metrics/short_seq_loss"], expected_crop.item(), places=5
            )

    def test_constant_batch_cyclic_lengths_match_direct_crop_updates(self):
        self._compare(dynamic_batch=False)

    def test_dynamic_batch_and_both_lengths_match_direct_crop_updates(self):
        self._compare(dynamic_batch=True)

    def _compare(self, dynamic_batch):
        trainer = dual_trainer(dynamic_batch=dynamic_batch)
        reference = tiny_trainer(False, k=2, b=4, seq=2, model=ScalarModel())
        reference.__class__ = Trainer
        reference.config.training.max_norm = 1e12
        loss_fn = reference.loss_fn
        reference.loss_fn = lambda p, y, n, **kw: loss_fn(p, y, n)
        iterator = iter(trainer.dataloader)
        total_grad = total_parent = 0
        signature = trainer._signature()
        for step in range(1, 13):
            trainer.step = reference.step = step
            plan = trainer._prepare_step()
            parents = [next(iterator) for _ in range(plan.loss_batches)]
            parents[0][1][0, -1] = -100
            crops = regroup_batches(trainer._crop(parents), plan.grad_microbatch)
            weight = trainer.model_parts[0].weight.detach()
            expected_long = sum(
                ((inputs["input"] * weight - y)[y != -100].square().sum()) for inputs, y in parents
            ) / sum(int((y != -100).sum()) for _, y in parents)
            calls = []
            handle = trainer.model_parts[0].register_forward_pre_hook(
                lambda model, args: calls.append((torch.is_grad_enabled(), tuple(args[0].shape)))
            )
            rng = torch.get_rng_state().clone()
            trainer.train_step(trainer.batch_generator(copy.deepcopy(parents)))
            handle.remove()
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            reference.gradient_accumulation_steps = plan.grad_batches
            reference.train_step(reference.batch_generator(copy.deepcopy(crops)))
            torch.testing.assert_close(
                trainer.optimizers.gradients[-1][0],
                reference.optimizers.gradients[-1][0],
                rtol=1e-12,
                atol=1e-8,
            )
            torch.testing.assert_close(
                trainer.model_parts[0].weight,
                reference.model_parts[0].weight,
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertEqual(
                calls,
                [(False, (plan.loss_microbatch, plan.loss_length))] * plan.loss_batches
                + [(True, (plan.grad_microbatch, plan.grad_length))] * plan.grad_batches,
            )
            self.assertEqual(len(trainer.optimizers.gradients), step)
            total_grad += plan.global_batch * plan.grad_length
            total_parent += plan.global_batch * plan.loss_length
            payload = trainer.metrics_processor.logger.log.call_args.args[0]
            self.assertAlmostEqual(
                payload["loss_metrics/long_avg_loss"], expected_long.item(), places=7
            )
            self.assertEqual(payload["n_tokens_seen"], total_grad)
            self.assertEqual(payload["loss_grad/parent_tokens_seen"], total_parent)
            self.assertEqual(payload["loss_grad/grad_microbatch_size"], plan.grad_microbatch)
            self.assertEqual(trainer._signature(), signature)

    def test_resume_at_batch_boundary_and_without_logging(self):
        trainer = dual_trainer(log_freq=4, dynamic_batch=True)
        iterator = trainer.batch_generator(trainer.dataloader)
        for step in range(1, 7):
            trainer.step = step
            trainer.train_step(iterator)
        restored = dual_trainer(log_freq=4, dynamic_batch=True)
        restored.model_parts[0].load_state_dict(copy.deepcopy(trainer.model_parts[0].state_dict()))
        restored.optimizers.optimizer.load_state_dict(
            copy.deepcopy(trainer.optimizers.optimizer.state_dict())
        )
        restored.lr_schedulers.scheduler.load_state_dict(
            trainer.lr_schedulers.scheduler.state_dict()
        )
        restored.dataloader.load_state_dict(copy.deepcopy(trainer.dataloader.state_dict()))
        restored.load_state_dict(trainer.state_dict())
        resumed = restored.batch_generator(restored.dataloader)
        for step in range(7, 13):
            trainer.step = restored.step = step
            trainer.train_step(iterator)
            restored.train_step(resumed)
            torch.testing.assert_close(
                trainer.model_parts[0].weight, restored.model_parts[0].weight, rtol=0, atol=0
            )
            self.assertEqual(trainer.state_dict(), restored.state_dict())
        restored.config.crop_seed += 1
        with self.assertRaisesRegex(ValueError, "differ"):
            restored.load_state_dict(trainer.state_dict())

    def test_long_forward_restores_rng_buffers_and_does_not_create_gradients(self):
        trainer = dual_trainer()
        trainer.model_parts[0].register_buffer("count", torch.zeros(()))

        def mutate(model, args):
            model.count.add_(1)
            torch.rand(10)

        handle = trainer.model_parts[0].register_forward_pre_hook(mutate)
        plan = trainer._prepare_step()
        parents = [next(iter(trainer.dataloader)) for _ in range(plan.loss_batches)]
        state = torch.get_rng_state().clone()
        trainer._long_forward(parents, trainer._check_data(parents))
        handle.remove()
        self.assertIsNone(trainer.model_parts[0].weight.grad)
        self.assertEqual(trainer.model_parts[0].count.item(), 0)
        self.assertEqual(trainer.ntokens_seen, 0)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)

    def test_partial_data_rejects_before_update(self):
        trainer = dual_trainer()
        state = trainer.state_dict()
        with self.assertRaises(DataloaderExhaustedError):
            trainer.train_step(iter([]))
        self.assertEqual(state, trainer.state_dict())
        self.assertEqual(len(trainer.optimizers.gradients), 0)
        with self.assertRaisesRegex(ValueError, "scheduled shape"):
            trainer.train_step(iter([({"input": torch.ones(1, 9)}, torch.ones(1, 9))]))
        self.assertEqual(state, trainer.state_dict())


class ConfigTest(unittest.TestCase):
    def test_yaml_profiles_and_formal_cycles(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "legacy/configs/loss_grad_seq").glob("*.yaml"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            raw, _, _, args = resolve_point(path, 0)
            with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
                cfg = ConfigManager().parse_args(
                    [
                        "--module",
                        "torchtitan_dynamic_seq_len.config_registry",
                        "--config",
                        raw["torchtitan_config"],
                        *args,
                    ]
                )
            schedules = cfg.validate_loss_grad()
            self.assertIs(cfg._owner, LossGradSeqTrainer)
            if path.stem == "formal_12288":
                seq = schedules["grad_seq_len_scheduler"]
                values = [seq.length_for_step(s) for s in range(1, 3841)]
                self.assertEqual(values, values[:320] * 12)
                self.assertEqual((min(values), max(values)), (512, 12288))
                self.assertEqual(set(values[s + 1] - values[s] for s in range(160)), {72, 76})
                self.assertEqual(sum(values), 3840 * 6400)
                self.assertEqual(raw["resources"]["account"], "jason_optai-paid")


if __name__ == "__main__":
    unittest.main()
