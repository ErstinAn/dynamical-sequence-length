"""Check interpolation, rounding, and checkpoint signatures."""

import unittest


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
