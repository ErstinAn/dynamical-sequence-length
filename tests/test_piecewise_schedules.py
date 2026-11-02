"""Check interpolation, complete experiment budgets, and optimizer LR endpoints."""

import unittest

import torch

from tests import test_dynamic_seq_bs as fixtures
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig


class PiecewiseScheduleTest(unittest.TestCase):
    def test_interpolation_rounding_repeat_and_hold(self):
        for repeat in (False, True):
            seq = SeqLenSchedulerConfig(
                kind="piecewise_linear",
                anchor_steps=[1, 4, 6],
                lengths=[8, 20, 8],
                round_to_multiple=4,
                repeat=repeat,
            ).build(12)
            batch = BatchSizeSchedulerConfig(
                kind="piecewise_linear",
                anchor_steps=[1, 4, 6],
                batch_sizes=[8, 20, 8],
                round_to_multiple=4,
                repeat=repeat,
            ).build(12)
            first = [8, 12, 16, 20, 16, 8]
            expected = first * 2 if repeat else first + [8] * 6
            self.assertEqual([seq.length_for_step(s) for s in range(1, 13)], expected)
            self.assertEqual([batch.value_for_step(s) for s in range(1, 13)], expected)
            self.assertEqual(seq.lengths, (8, 12, 16, 20))
            with self.assertRaises(ValueError):
                seq.length_for_step(0)
        plateau = SeqLenSchedulerConfig(
            kind="piecewise_linear", anchor_steps=[1, 3, 4, 6], lengths=[8, 8, 16, 16]
        ).build(6)
        self.assertEqual([plateau.length_for_step(s) for s in range(1, 7)], [8, 8, 8, 16, 16, 16])

    def test_invalid_anchors_and_rounding(self):
        defaults = dict(kind="piecewise_linear", anchor_steps=[1, 4], lengths=[8, 16])
        for change in (
            dict(anchor_steps=[]),
            dict(anchor_steps=[2, 4]),
            dict(anchor_steps=[1, 1]),
            dict(anchor_steps=[1, 3, 2]),
            dict(anchor_steps=[1, 7]),
            dict(anchor_steps=[1, 4.0]),
            dict(lengths=[8]),
            dict(lengths=[0, 16]),
            dict(lengths=[8, True]),
            dict(round_to_multiple=0),
            dict(round_to_multiple=3),
            dict(repeat=1),
            dict(boundaries=[2]),
            dict(phase_offset_steps=1),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                SeqLenSchedulerConfig(**(defaults | change)).build(6)
        with self.assertRaisesRegex(ValueError, "require a piecewise_linear"):
            SeqLenSchedulerConfig(kind="constant", anchor_steps=[1, 4]).build(6)

    def test_profiles_have_equal_steps_and_total_tokens(self):
        profiles = {}
        for filename in (
            "linear_bs_cyclic_seq_cosine_lr.yaml",
            "linear_bs_cyclic_seq_constant_lr.yaml",
            "four_stage_cycle.yaml",
        ):
            for world in (2, 4):
                with self.subTest(filename=filename, world=world):
                    cfg = fixtures.SchedulerTest().config(filename, world)
                    seq = cfg.validate_dynamic()
                    batch = cfg.batch_size_scheduler.build(cfg.training.steps)
                    micro = cfg.microbatch_scheduler.build(cfg.training.steps)
                    rows = [
                        (seq.length_for_step(s), batch.value_for_step(s), micro.value_for_step(s))
                        for s in range(1, cfg.training.steps + 1)
                    ]
                    self.assertEqual(len(rows), 3840)
                    self.assertEqual(sum(length * b for length, b, _ in rows), 2013265920)
                    self.assertTrue(all(b == m * 8 for _, b, m in rows))
                    self.assertEqual({b // (world * m) for _, b, m in rows}, {8 // world})
                    self.assertEqual(cfg.validator.eval_seq_len, 2048)
                    self.assertEqual(cfg.validator.freq, 20)
                    profiles[filename] = rows
        rows = profiles["linear_bs_cyclic_seq_cosine_lr.yaml"]
        self.assertEqual(rows, profiles["linear_bs_cyclic_seq_constant_lr.yaml"])
        self.assertEqual((rows[0][1], rows[-1][1]), (256, 768))
        self.assertTrue(all(a[1] <= b[1] for a, b in zip(rows, rows[1:])))
        lengths = [row[0] for row in rows]
        self.assertEqual(lengths, lengths[:256] * 15)
        self.assertEqual(lengths, lengths[::-1])
        self.assertEqual((min(lengths), max(lengths)), (8, 2040))
        self.assertEqual(lengths[:128], list(range(8, 2041, 16)))
        self.assertEqual(lengths[128:256], list(range(2040, 7, -16)))
        self.assertTrue(all(a[1] + b[1] == 1024 for a, b in zip(rows, rows[::-1])))
        cycle = profiles["four_stage_cycle.yaml"]
        self.assertEqual(cycle, cycle[:960] * 4)
        self.assertEqual(cycle[:240], [(512, 256, 32)] * 240)
        self.assertEqual(cycle[480:720], [(1472, 1024, 128)] * 240)
        self.assertEqual(cycle[720:960], [(512, 256, 32)] * 240)
        self.assertEqual(cycle[240], (512, 256, 32))
        self.assertEqual(cycle[479], (512, 1024, 128))
        self.assertTrue(all(row[0] == 512 for row in cycle[240:480]))
        self.assertTrue(all(a[1] <= b[1] for a, b in zip(cycle[240:479], cycle[241:480])))

    def test_actual_optimizer_lr_and_resume(self):
        for name in ("cosine", "constant"):
            cfg = fixtures.SchedulerTest().config(f"linear_bs_cyclic_seq_{name}_lr.yaml")

            def build():
                opt = torch.optim.SGD(
                    [torch.nn.Parameter(torch.zeros(1))], lr=cfg.optimizer.learning_rate
                )
                return opt, cfg.lr_scheduler.build(
                    optimizers=[opt], training_steps=cfg.training.steps
                )

            opt, sched = build()
            resumed = None
            rates = []
            for step in range(1, 3841):
                rates.append(opt.param_groups[0]["lr"])
                if resumed:
                    self.assertEqual(opt.param_groups[0]["lr"], resumed[0].param_groups[0]["lr"])
                    resumed[0].step()
                    resumed[1].step()
                opt.step()
                sched.step()
                if step == 1777:
                    resumed = build()
                    resumed[0].load_state_dict(opt.state_dict())
                    resumed[1].load_state_dict(sched.state_dict())
            self.assertEqual(rates[0], 5e-4)
            if name == "cosine":
                self.assertEqual(rates[-1], 0.0)
                self.assertTrue(all(a >= b >= 0 for a, b in zip(rates, rates[1:])))
            else:
                self.assertEqual(set(rates), {5e-4})
        cfg = fixtures.SchedulerTest().config("four_stage_cycle.yaml")
        self.assertEqual(cfg.lr_scheduler.min_lr_factor, 1.0)
        self.assertEqual(cfg.lr_scheduler.warmup_steps, 0)

    def test_anchor_changes_invalidate_checkpoint_signature(self):
        trainer = fixtures.bs_trainer()
        trainer.config.batch_size_scheduler = BatchSizeSchedulerConfig(
            kind="piecewise_linear", anchor_steps=[1, 6], batch_sizes=[4, 8]
        )
        original = trainer._schedule_signature()
        trainer.config.batch_size_scheduler.anchor_steps[-1] = 5
        self.assertNotEqual(trainer._schedule_signature(), original)
        trainer.config.batch_size_scheduler.anchor_steps[-1] = 6
        trainer.config.batch_size_scheduler.repeat = True
        self.assertNotEqual(trainer._schedule_signature(), original)


if __name__ == "__main__":
    unittest.main()
