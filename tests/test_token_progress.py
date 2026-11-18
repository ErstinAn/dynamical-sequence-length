"""Check token thresholds, unchanged loss keys, validation axes, and complete recovery."""

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torchtitan.components.metrics import BaseLogger, LoggerContainer, WandBLogger
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.models.llama3 import model_registry

from tests.test_dynamic_seq_bs import bs_trainer
from tests.test_fixed_validation import _Model, _validator, _write_artifact
from torchtitan_dynamic_seq_len.experiment import ExperimentValidator, SequenceExperimentTrainer, llama3_1b
from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers
from torchtitan_dynamic_seq_len.token_metrics import LOSS_KEYS, TokenMetricsProcessor, define_token_axes
from torchtitan_dynamic_seq_len.token_progress import TokenCheckpointManager, TokenPlan


def token_trainer(log_freq=4, steps=20, budget=570, validation_intervals=0):
    trainer = bs_trainer(log_freq=log_freq)
    trainer.__class__ = SequenceExperimentTrainer
    trainer.config.training.steps = steps
    trainer.seq_len_scheduler = trainer.config.seq_len_scheduler.build(steps)
    trainer.batch_size_scheduler = trainer.config.batch_size_scheduler.build(steps)
    trainer.microbatch_scheduler = trainer.config.microbatch_scheduler.build(steps)
    trainer.token_plan = TokenPlan.build([8, 8, 16] + [32] * (steps - 3), budget, validation_intervals)
    trainer.validation_by_step = {point.step: point for point in trainer.token_plan.validations}
    trainer.metrics_processor.__class__ = TokenMetricsProcessor
    trainer.metrics_processor.set_training_tokens(0)
    trainer.optimizers = ExperimentOptimizers.Config(learning_rate=0.03).build(
        model_parts=trainer.model_parts)
    trainer.lr_schedulers = LRSchedulersContainer.Config(warmup_steps=0).build(
        optimizers=trainer.optimizers, training_steps=steps)
    return trainer


def checkpoint_manager(trainer, root):
    return TokenCheckpointManager.Config().build(
        dataloader=trainer.dataloader, model_parts=trainer.model_parts,
        optimizers=trainer.optimizers,
        lr_schedulers=trainer.lr_schedulers,
        states={"train_state": trainer}, sd_adapter=None, base_folder=str(root))


class TokenPlanTest(unittest.TestCase):
    def test_eight_step_smoke_uses_phase_checkpoints_without_changing_formal_policy(self):
        lengths = [2048, 2048, 4096, 4096, 8192, 8192, 16384, 16384]
        plan = TokenPlan.smoke([512 * length for length in lengths], 31457280)
        self.assertEqual([item.step for item in plan.milestones], [2, 4, 6, 8])
        self.assertEqual(plan.cumulative[4], 6291456)
        self.assertEqual(plan.cumulative[-1], 31457280)
        with self.assertRaisesRegex(ValueError, "multiple checkpoint"):
            TokenPlan.build([512 * length for length in lengths], 31457280)

    def test_variable_steps_and_integer_thresholds(self):
        plan = TokenPlan.build([8, 8, 16] + [32] * 17, 570)
        self.assertEqual([m.step for m in plan.milestones], [4, 6, 8, 10, 11, 13, 15, 17, 19, 20])
        self.assertEqual([m.threshold for m in plan.milestones], list(range(57, 571, 57)))
        self.assertEqual(plan.milestones[-1].tokens, 576)
        self.assertEqual(plan.milestones[-1].excess, 6)
        large = 2**54 + 1
        plan = TokenPlan.build([large] * 10, large * 10)
        self.assertEqual([m.threshold for m in plan.milestones], [large * i for i in range(1, 11)])

    def test_rejects_incomplete_repeated_or_colliding_milestones(self):
        for increments, budget in [([10] * 9, 100), ([10] * 11, 100), ([100], 100),
                                   ([1] * 10, 0), ([0] * 10, 10), ([True] * 10, 10)]:
            with self.subTest(increments=increments, budget=budget), self.assertRaises(ValueError):
                TokenPlan.build(increments, budget)
        with self.assertRaisesRegex(ValueError, "multiple validation"):
            TokenPlan.build([1] * 90 + [9] * 10, 180, validation_intervals=50)

    def test_rejects_retention_and_partial_recovery(self):
        for key, value in [("keep_latest_k", 10), ("enable", False),
                           ("last_save_model_only", True), ("async_mode", "async"),
                           ("exclude_from_loading", ["optimizer"]),
                           ("enable_first_step_checkpoint", True), ("create_seed_checkpoint", True)]:
            config = TokenCheckpointManager.Config()
            setattr(config, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                config.validate_policy()


class TokenMetricsTest(unittest.TestCase):
    def test_wandb_keeps_names_and_binds_explicit_token_axes(self):
        define_token_axes(BaseLogger())  # Ranks without a logger must also initialize.
        for tag in (None, "replica"):
            container = LoggerContainer()
            logger = object.__new__(WandBLogger)
            logger.tag, logger.wandb = tag, SimpleNamespace(run=Mock())
            container.add_logger(logger)
            define_token_axes(container)
            prefix = "" if tag is None else tag + "/"
            self.assertEqual(logger.wandb.run.define_metric.call_count, 4)
            for key in LOSS_KEYS:
                logger.wandb.run.define_metric.assert_any_call(
                    prefix + key, step_metric=prefix + "n_tokens_seen", step_sync=False)

    def test_validation_between_logs_uses_current_total_including_masked_positions(self):
        trainer = token_trainer()
        iterator = iter(trainer.dataloader)
        for step in (1, 2):
            trainer.step = step
            trainer._prepare_step()
            batches = [next(iterator) for _ in range(trainer.gradient_accumulation_steps)]
            batches[0][1][0, 0] = -100
            trainer.train_step(trainer.batch_generator(batches))
        self.assertEqual(trainer.nominal_tokens_seen, 16)
        self.assertEqual(trainer.n_valid_tokens_seen, 14)
        training = trainer.metrics_processor.logger.log.call_args.args[0]
        self.assertEqual(training["n_tokens_seen"], 8)
        self.assertEqual({k for k in training if "loss" in k}, set(LOSS_KEYS[:2]))
        before = copy.deepcopy(trainer.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_artifact(root)
            validator = _validator(root)
            validator.metrics_processor = trainer.metrics_processor
            validator.validate([_Model()], step=2)
        validation = trainer.metrics_processor.logger.log.call_args.args[0]
        self.assertEqual(validation["n_tokens_seen"], 16)
        self.assertEqual({k for k in validation if "loss" in k}, {LOSS_KEYS[2]})
        self.assertEqual(validation["validation_metrics/evaluated_tokens"], 96)
        self.assertEqual(before, trainer.state_dict())


class TokenCheckpointTest(unittest.TestCase):
    def test_validation_from_initial_model_to_final_step_across_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_artifact(root)

            def build():
                trainer = token_trainer(steps=60, budget=1856, validation_intervals=50)
                trainer.config.dump_folder = folder
                trainer.validator = _validator(root, validator_class=ExperimentValidator, token_intervals=50)
                trainer.validator.metrics_processor = trainer.metrics_processor
                trainer.validator.loss_fn = lambda pred, labels: ((pred - labels).square().sum(), {})
                trainer.checkpointer = checkpoint_manager(trainer, root)
                self.addCleanup(trainer.checkpointer.close)
                return trainer

            def run(trainer, first, last):
                iterator = trainer.batch_generator(trainer.dataloader)
                for step in range(first, last + 1):
                    trainer.step = step
                    trainer.train_step(iterator)
                    self.assertFalse(trainer.validator.should_validate(step))
                    trainer.checkpointer.save(step, last_step=step == 60)

            original = build()
            expected_initial = original.model_parts[0].weight.detach().item()
            first_save = original.token_plan.milestones[0].step
            run(original, 1, first_save + 2)
            baseline = json.loads((root / "validation/point-000.json").read_text())
            inputs = torch.arange(96).double() % 7
            expected_loss = ((inputs * expected_initial - (inputs + 1) % 7) ** 2).mean().item()
            self.assertAlmostEqual(baseline["loss"], expected_loss)

            resumed = build()
            self.assertTrue(resumed.checkpointer.load())
            self.assertEqual(resumed.step, first_save)
            run(resumed, resumed.step + 1, 60)
            logs = [call.args[0] for trainer in (original, resumed)
                    for call in trainer.metrics_processor.logger.log.call_args_list]
            points = [row for row in logs if "validation_metrics/index" in row]
            plan = original.token_plan
            self.assertEqual([row["validation_metrics/index"] for row in points], list(range(51)))
            self.assertEqual([p.step for p in plan.validations[5::5]], [m.step for m in plan.milestones])
            for row, point in zip(points, plan.validations, strict=True):
                self.assertEqual(row["n_tokens_seen"], point.tokens)
                self.assertEqual(row["validation_metrics/completed_updates"], point.step)
                self.assertEqual(row["validation_metrics/target_tokens"], (plan.budget * point.index + 49) // 50)
                self.assertEqual(row["validation_metrics/evaluated_tokens"], 96)
                receipt = json.loads((root / f"validation/point-{point.index:03d}.json").read_text())
                self.assertEqual(receipt["step"], point.step)
                self.assertEqual(receipt["loss"], row["validation_metrics/loss"])
            self.assertEqual(resumed.nominal_tokens_seen, plan.budget)
            self.assertEqual(len(list((root / "checkpoint").glob("step-*"))), 10)

    def test_ten_complete_saves_and_resume_without_duplicate_milestones(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = token_trainer()
            trainer.checkpointer = checkpoint_manager(trainer, root)
            self.addCleanup(trainer.checkpointer.close)
            iterator = trainer.batch_generator(trainer.dataloader)
            for step in range(1, 11):
                trainer.step = step
                trainer.train_step(iterator)
                trainer.checkpointer.save(step)
            resumed = token_trainer()
            resumed.checkpointer = checkpoint_manager(resumed, root)
            self.addCleanup(resumed.checkpointer.close)
            self.assertTrue(resumed.checkpointer.load())
            self.assertEqual(resumed.step, 10)
            self.assertEqual(resumed.metrics_processor.training_tokens, 256)
            restored = resumed.batch_generator(resumed.dataloader)
            for step in range(11, 21):
                trainer.step = resumed.step = step
                trainer.train_step(iterator)
                resumed.train_step(restored)
                torch.testing.assert_close(trainer.model_parts[0].weight,
                                           resumed.model_parts[0].weight, rtol=0, atol=0)
                self.assertEqual(trainer.state_dict(), resumed.state_dict())
                resumed.checkpointer.save(step, last_step=step == 20)
            folders = sorted((root / "checkpoint").glob("step-*"))
            self.assertEqual(len(folders), 10)
            self.assertFalse((root / "checkpoint/step-0").exists())
            for item in resumed.token_plan.milestones:
                path = root / "checkpoint" / f"step-{item.step}"
                self.assertTrue((path / ".metadata").is_file())
                receipt = json.loads((path / "token_milestone.json").read_text())
                self.assertEqual((receipt["index"], receipt["tokens"]), (item.index, item.tokens))
            final = token_trainer()
            final.checkpointer = checkpoint_manager(final, root)
            self.addCleanup(final.checkpointer.close)
            self.assertTrue(final.checkpointer.load())
            self.assertEqual(final.state_dict(), resumed.state_dict())
            self.assertTrue(final.optimizers.state_dict())
            self.assertEqual(final.lr_schedulers.state_dict(), resumed.lr_schedulers.state_dict())

    def test_recovery_rejects_changed_budget_and_corrupt_counts(self):
        trainer = token_trainer()
        state = trainer.state_dict()
        for key, value in [("token_budget", 571), ("nominal_tokens_seen", 1), ("ntokens_seen", 1)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "budget or cumulative"):
                trainer.load_state_dict({**state, key: value})


class MainEntryTest(unittest.TestCase):
    def test_model_definition_matches_native_1b_and_data_protocol_is_fixed(self):
        config = llama3_1b()
        native = model_registry("1B", attn_backend="varlen")
        def without_function_addresses(model):
            return re.sub(r" at 0x[0-9a-f]+", "", json.dumps(model.to_dict(), sort_keys=True))
        self.assertEqual(without_function_addresses(config.model_spec.model),
                         without_function_addresses(native.model))
        self.assertEqual(config.model_spec.model.vocab_size, 128256)
        self.assertEqual(config.dataloader.packing, "stream")
        self.assertFalse(config.dataloader.infinite)
        self.assertEqual(config.validator.eval_seq_len, 16384)
        self.assertEqual(config.validator.token_budget, 15335424)
        self.assertTrue(config.metrics.enable_wandb)
        with self.assertRaises(ValueError):
            config.validate_dynamic()


if __name__ == "__main__":
    unittest.main()
