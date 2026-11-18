"""Check independent schedules, variable batches, gradients, and recovery."""

import copy
import os
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from torchtitan.config import ConfigManager
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.trainer import Trainer

from tests.test_dynamic_seq import IntegerTokenizer, ScalarModel, dynamic_trainer
from tests.test_grad_var_trainer import tiny_trainer
from torchtitan_dynamic_seq_len.data.dynamic_seq_bs_loader import DynamicSeqBSDataLoader
from torchtitan_dynamic_seq_len.dynamic_seq_bs.metrics import DynamicSeqBSMetricsProcessor
from torchtitan_dynamic_seq_len.dynamic_seq_bs.trainer import DynamicSeqBSTrainer
from torchtitan_dynamic_seq_len.run_config import resolve_point
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig


def batch_loader(*, streaming=True, infinite=False, documents=None, dp_rank=0, dp_world_size=1):
    documents = documents or [" ".join(map(str, range(2001)))]
    dataset = Dataset.from_dict({"text": documents})
    if streaming:
        dataset = dataset.to_iterable_dataset()
    DATASETS["dynamic_bs_test"] = DatasetConfig(
        path="unused", loader=lambda path: dataset, sample_processor=lambda row: row["text"]
    )
    return DynamicSeqBSDataLoader(
        DynamicSeqBSDataLoader.Config(
            dataset="dynamic_bs_test", infinite=infinite, pin_memory=False
        ),
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=IntegerTokenizer(),
        seq_len=8,
        local_batch_size=2,
        snapshot_every_n_steps=500,
    )


def bs_trainer(log_freq=1):
    trainer = dynamic_trainer(log_freq=log_freq)
    trainer.__class__ = DynamicSeqBSTrainer
    trainer.config.batch_size_scheduler = BatchSizeSchedulerConfig(
        kind="increasing", batch_sizes=[4, 8], boundaries=[3]
    )
    trainer.config.microbatch_scheduler = BatchSizeSchedulerConfig(
        kind="decreasing", batch_sizes=[1, 2], boundaries=[3]
    )
    trainer.batch_size_scheduler = trainer.config.batch_size_scheduler.build(6)
    trainer.microbatch_scheduler = trainer.config.microbatch_scheduler.build(6)
    trainer.config.training.max_norm = 1e12
    trainer.metrics_processor.__class__ = DynamicSeqBSMetricsProcessor
    trainer.dataloader = batch_loader()
    return trainer


class SchedulerTest(unittest.TestCase):
    def test_all_kinds_share_sequence_schedule_rules(self):
        for kind in ("constant", "increasing", "decreasing", "triangular", "cyclic"):
            with self.subTest(kind=kind):
                batch = BatchSizeSchedulerConfig(
                    kind=kind,
                    batch_size=8,
                    batch_sizes=[4, 8, 12],
                    period_steps=4,
                    min_batch_size=4,
                    max_batch_size=12,
                ).build(12)
                seq = SeqLenSchedulerConfig(
                    kind=kind,
                    length=8,
                    lengths=[4, 8, 12],
                    period_steps=4,
                    min_length=4,
                    max_length=12,
                ).build(12)
                self.assertEqual(
                    [batch.value_for_step(s) for s in range(1, 13)],
                    [seq.length_for_step(s) for s in range(1, 13)],
                )
                with self.assertRaises(ValueError):
                    batch.value_for_step(0)
        rounded = BatchSizeSchedulerConfig(
            kind="cyclic",
            min_batch_size=8,
            max_batch_size=2040,
            period_steps=256,
            round_to_multiple=8,
        ).build(256)
        self.assertEqual(sum(rounded.value_for_step(s) for s in range(1, 257)), 256 * 1024)

    def config(self, filename="smoke.yaml", world=2):
        path = Path(__file__).resolve().parents[1] / "legacy/configs/dynamic_seq_bs" / filename
        raw, _, _, args = resolve_point(path, 0)
        with patch.dict(os.environ, {"WORLD_SIZE": str(world)}):
            return ConfigManager().parse_args(
                [
                    "--module",
                    "torchtitan_dynamic_seq_len.config_registry",
                    "--config",
                    raw["torchtitan_config"],
                    *args,
                ]
            )

    def test_public_profiles_and_owners(self):
        for filename, expected in [("smoke.yaml", 6400), ("increasing.yaml", 1761607680)]:
            for world in (1, 2, 4):
                with self.subTest(filename=filename, world=world):
                    cfg = self.config(filename, world)
                    seq = cfg.validate_dynamic()
                    bs = cfg.batch_size_scheduler.build(cfg.training.steps)
                    self.assertIs(cfg._owner, DynamicSeqBSTrainer)
                    self.assertIs(cfg.dataloader._owner, DynamicSeqBSDataLoader)
                    self.assertIs(cfg.metrics._owner, DynamicSeqBSMetricsProcessor)
                    self.assertEqual(
                        sum(
                            seq.length_for_step(s) * bs.value_for_step(s)
                            for s in range(1, cfg.training.steps + 1)
                        ),
                        expected,
                    )

    def test_rejects_invalid_intermediate_step_and_initial_values(self):
        cfg = self.config()
        cfg.batch_size_scheduler = BatchSizeSchedulerConfig(
            kind="cyclic",
            min_batch_size=8,
            max_batch_size=12,
            period_steps=8,
        )
        cfg.microbatch_scheduler = BatchSizeSchedulerConfig(batch_size=2)
        with self.assertRaisesRegex(ValueError, "step 2:.*global_batch=9"):
            cfg.validate_dynamic()
        for attr, value in [("global_batch_size", 16), ("local_batch_size", 4)]:
            cfg = self.config()
            setattr(cfg.training, attr, value)
            with self.assertRaisesRegex(ValueError, "step 1"):
                cfg.validate_dynamic()
        cfg = self.config()
        cfg.dataloader.num_workers = 1
        with self.assertRaisesRegex(ValueError, "num_workers=0"):
            cfg.validate_dynamic()

    def test_fixed_token_profiles(self):
        schedules = {}
        for kind in ("increasing", "decreasing", "cyclic"):
            for world in (2, 4):
                with self.subTest(kind=kind, world=world):
                    cfg = self.config(f"fixed_tokens_bs_{kind}.yaml", world)
                    seq = cfg.validate_dynamic()
                    bs = cfg.batch_size_scheduler.build(cfg.training.steps)
                    micro = cfg.microbatch_scheduler.build(cfg.training.steps)
                    values = [
                        (seq.length_for_step(s), bs.value_for_step(s), micro.value_for_step(s))
                        for s in range(1, cfg.training.steps + 1)
                    ]
                    self.assertEqual(len(values), 3840)
                    self.assertEqual({length * batch for length, batch, _ in values}, {524288})
                    self.assertEqual({length * m for length, _, m in values}, {131072})
                    self.assertEqual({batch // (world * m) for _, batch, m in values}, {4 // world})
                    self.assertEqual(sum(length * batch for length, batch, _ in values), 2013265920)
                    self.assertEqual(
                        {length for length, _, _ in values}, {8 * 2**i for i in range(9)}
                    )
                    self.assertEqual(cfg.validator.eval_seq_len, 2048)
                    self.assertEqual(cfg.validator.freq, 20)
                    self.assertEqual(cfg.training.seq_len, 2048)
                    self.assertEqual(cfg.optimizer.learning_rate, 5e-4)
                    self.assertEqual(cfg.lr_scheduler.warmup_steps, 0)
                    self.assertEqual(cfg.lr_scheduler.min_lr_factor, 1.0)
                    schedules[kind] = values
        self.assertEqual(schedules["decreasing"], schedules["increasing"][::-1])
        increasing = schedules["increasing"]
        durations = [240, 480, 480, 480, 480, 480, 480, 480, 240]
        expected = []
        for i, duration in enumerate(durations):
            expected.extend([(2048 // 2**i, 256 * 2**i, 64 * 2**i)] * duration)
        self.assertEqual(increasing, expected)
        cyclic = schedules["cyclic"]
        self.assertEqual(Counter(cyclic), Counter(increasing))
        self.assertEqual(Counter(cyclic), Counter(schedules["decreasing"]))
        self.assertEqual(cyclic, cyclic[:256] * 15)
        for phase, index in enumerate([*range(9), *range(7, 0, -1)]):
            self.assertEqual(
                cyclic[phase * 16 : (phase + 1) * 16],
                [(2048 // 2**index, 256 * 2**index, 64 * 2**index)] * 16,
            )

    def test_triangular_phase_offset(self):
        for config, field in (
            (SeqLenSchedulerConfig, "length_for_step"),
            (BatchSizeSchedulerConfig, "value_for_step"),
        ):
            base = config(kind="triangular", period_steps=16).build(32)
            shifted = config(kind="triangular", period_steps=16, phase_offset_steps=8).build(32)
            self.assertEqual(
                [getattr(shifted, field)(s) for s in range(1, 33)],
                [getattr(base, field)(s + 8) for s in range(1, 33)],
            )
            for offset in (-1, 0.5, True, 16):
                with self.subTest(config=config, offset=offset), self.assertRaises(ValueError):
                    config(kind="triangular", period_steps=16, phase_offset_steps=offset).build(32)
            with self.assertRaisesRegex(ValueError, "requires a triangular"):
                config(kind="constant", phase_offset_steps=1).build(32)


class LoaderTest(unittest.TestCase):
    def test_changes_shapes_without_losing_tokens_and_restores(self):
        for streaming in (False, True):
            dl = batch_loader(streaming=streaming)
            self.assertEqual(dl.snapshot_every_n_steps, 1)
            it = iter(dl)
            offset = 0
            for index, (length, micro) in enumerate(
                [(2, 4), (8, 1), (4, 3), (2, 1), (8, 4), (4, 2)]
            ):
                if index == 3:
                    saved = copy.deepcopy(dl.state_dict())
                    dl = batch_loader(streaming=streaming)
                    dl.load_state_dict(saved)
                    it = iter(dl)
                dl.set_batch_shape(seq_len=length, microbatch_size=micro)
                inputs, labels = next(it)
                self.assertEqual(labels.shape, (micro, length))
                torch.testing.assert_close(
                    inputs["input"].flatten(), torch.arange(offset, offset + micro * length)
                )
                torch.testing.assert_close(labels, inputs["input"] + 1)
                self.assertTrue(bool((inputs["positions"] < length).all()))
                offset += micro * length

    def test_partial_batch_and_exact_end_resume(self):
        dl = batch_loader(documents=["0 1 2 3 4 5 6"])
        dl.set_batch_shape(seq_len=2, microbatch_size=2)
        it = iter(dl)
        self.assertEqual(next(it)[1].shape, (2, 2))
        self.assertEqual(next(it)[1].shape, (1, 2))
        with self.assertRaises(StopIteration):
            next(it)
        dl = batch_loader(documents=["0 1 2 3 4"], infinite=True)
        dl.set_batch_shape(seq_len=2, microbatch_size=2)
        next(iter(dl))
        saved = copy.deepcopy(dl.state_dict())
        restored = batch_loader(documents=["0 1 2 3 4"], infinite=True)
        restored.load_state_dict(saved)
        restored.set_batch_shape(seq_len=4, microbatch_size=1)
        self.assertEqual(next(iter(restored))[1].tolist(), [[1, 2, 3, 4]])


class TrainerTest(unittest.TestCase):
    def test_phase_offset_checkpoint_signatures(self):
        def legacy_asdict(config):
            result = asdict(config)
            result.pop("phase_offset_steps", None)
            result.pop("anchor_steps", None)
            result.pop("repeat", None)
            return result

        for factory, module in (
            (dynamic_trainer, "torchtitan_dynamic_seq_len.dynamic_seq.trainer"),
            (bs_trainer, "torchtitan_dynamic_seq_len.dynamic_seq_bs.trainer"),
        ):
            trainer = factory()
            with patch(f"{module}.asdict", side_effect=legacy_asdict):
                previous = trainer._schedule_signature()
            self.assertEqual(trainer._schedule_signature(), previous)
            trainer.config.seq_len_scheduler.phase_offset_steps = 2
            self.assertNotEqual(trainer._schedule_signature(), previous)

    def test_native_gradient_equivalence_and_metrics(self):
        trainer = bs_trainer()
        native = tiny_trainer(False, k=1, b=4, seq=8, model=ScalarModel())
        native.__class__ = Trainer
        native.config.training.max_norm = 1e12
        loss = native.loss_fn
        native.loss_fn = lambda pred, labels, total, **kwargs: loss(pred, labels, total)
        native.dataloader = batch_loader()
        iterator = trainer.batch_generator(trainer.dataloader)
        reference = native.batch_generator(native.dataloader)
        total = 0
        signature = trainer._schedule_signature()
        for step in range(1, 7):
            trainer.step = native.step = step
            length = trainer.seq_len_scheduler.length_for_step(step)
            global_bs = trainer.batch_size_scheduler.value_for_step(step)
            native.dataloader.set_batch_shape(seq_len=length, microbatch_size=global_bs)
            trainer.train_step(iterator)
            native.train_step(reference)
            torch.testing.assert_close(
                trainer.optimizers.gradients[-1][0],
                native.optimizers.gradients[-1][0],
                rtol=1e-12,
                atol=1e-9,
            )
            torch.testing.assert_close(
                trainer.model_parts[0].weight, native.model_parts[0].weight, rtol=1e-12, atol=1e-12
            )
            total += global_bs * length
            payload = trainer.metrics_processor.logger.log.call_args.args[0]
            self.assertEqual(payload["valid_tokens_per_step"], global_bs * length)
            self.assertEqual(payload["n_tokens_seen"], total)
            self.assertEqual(payload["n_valid_tokens_seen"], total)
            self.assertEqual(payload["global_batch_size"], global_bs)
            self.assertEqual(
                payload["microbatch_size"] * payload["gradient_accumulation_steps"], global_bs
            )
            keys = list(payload)
            self.assertEqual(
                keys[keys.index("lr") + 1 : keys.index("lr") + 5],
                ["seq_len", "global_batch_size", "microbatch_size", "gradient_accumulation_steps"],
            )
            self.assertEqual(trainer._schedule_signature(), signature)
        self.assertEqual(trainer.config.training.local_batch_size, 2)
        self.assertEqual(trainer.config.training.global_batch_size, 4)

    def test_resume_at_batch_transition_and_nonlogging_step(self):
        original = bs_trainer(log_freq=4)
        iterator = original.batch_generator(original.dataloader)
        for step in (1, 2, 3):
            original.step = step
            original.train_step(iterator)
        resumed = bs_trainer(log_freq=4)
        resumed.model_parts[0].load_state_dict(copy.deepcopy(original.model_parts[0].state_dict()))
        resumed.optimizers.optimizer.load_state_dict(
            copy.deepcopy(original.optimizers.optimizer.state_dict())
        )
        resumed.lr_schedulers.scheduler.load_state_dict(
            original.lr_schedulers.scheduler.state_dict()
        )
        resumed.dataloader.load_state_dict(copy.deepcopy(original.dataloader.state_dict()))
        resumed.load_state_dict(original.state_dict())
        ri = resumed.batch_generator(resumed.dataloader)
        for step in (4, 5, 6):
            original.step = resumed.step = step
            original.train_step(iterator)
            resumed.train_step(ri)
            torch.testing.assert_close(
                original.model_parts[0].weight, resumed.model_parts[0].weight, rtol=0, atol=0
            )
            self.assertEqual(original.state_dict(), resumed.state_dict())
        resumed.config.microbatch_scheduler.batch_sizes = [1, 4]
        with self.assertRaisesRegex(ValueError, "differs"):
            resumed.load_state_dict(original.state_dict())

    def test_window_flops_use_actual_tokens(self):
        trainer = bs_trainer(log_freq=6)
        it = trainer.batch_generator(trainer.dataloader)
        expected_tokens = expected_flops = 0
        for step in range(1, 7):
            trainer.step = step
            trainer.train_step(it)
            length = trainer.current_seq_len
            tokens = trainer.current_global_batch_size * length
            if step > 1:
                expected_tokens += tokens
                expected_flops += tokens * trainer.flops_by_length[length]
        self.assertEqual(
            trainer.metrics_processor.num_flops_per_token, expected_flops / expected_tokens
        )


if __name__ == "__main__":
    unittest.main()
