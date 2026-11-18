"""Check experiment trends, rounding, cycle counts, and token accounting."""

import unittest

from scripts.prepare_longseq_joint_configs import compressed_schedule
from scripts.prepare_longseq_loss_grad_configs import NAMES, STEPS, TARGET, make_schedule
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig


class ScheduleTests(unittest.TestCase):
    def test_budgets_and_batch_serialization(self):
        for name in NAMES.values():
            with self.subTest(name=name):
                _, lengths, batches = make_schedule(name)
                self.assertEqual((len(lengths), min(lengths), max(lengths)), (STEPS, 1024, 12288))
                self.assertTrue(all(g % 8 == b % 8 == 0 for g, b in zip(lengths, batches)))
                total = sum(g * b for g, b in zip(lengths, batches))
                self.assertLess(abs(total / TARGET - 1), 0.0004)
                self.assertLess(sum(batches) * 12288, 33_000_000_000)
                schedule = BatchSizeSchedulerConfig(**compressed_schedule(batches)).build(STEPS)
                self.assertEqual(batches, [schedule.value_for_step(s) for s in range(1, STEPS + 1)])

    def test_monotonic_mirrors(self):
        _, up, down_batch = make_schedule("bs_decreasing_seq_increasing")
        _, down, up_batch = make_schedule("bs_increasing_seq_decreasing")
        self.assertEqual(up, down[::-1])
        self.assertEqual(down_batch, up_batch[::-1])
        self.assertTrue(all(a <= b for a, b in zip(up, up[1:])))
        self.assertTrue(all(a >= b for a, b in zip(down_batch, down_batch[1:])))

    def test_cycles_and_linear_batch_pair(self):
        for name in ("bs_cyclic_seq_inverse", "linear_bs_cyclic_seq_constant_lr", "four_stage_cycle"):
            _, lengths, batches = make_schedule(name)
            self.assertEqual(lengths, lengths[:320] * 12)
            if name != "linear_bs_cyclic_seq_constant_lr":
                self.assertEqual(batches, batches[:320] * 12)
        _, a, b = make_schedule("linear_bs_cyclic_seq_constant_lr")
        _, c, d = make_schedule("linear_bs_cyclic_seq_cosine_lr")
        self.assertEqual((a, b), (c, d))
        self.assertTrue(all(x <= y for x, y in zip(b, b[1:])))


if __name__ == "__main__":
    unittest.main()
