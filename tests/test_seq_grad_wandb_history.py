import unittest

from scripts.seq_grad_wandb_history import scan_metric_rows


class History:
    def __init__(self, rows):
        self.rows = rows

    def scan_history(self, keys):
        return iter(self.rows)


class MetricHistoryTests(unittest.TestCase):
    def test_dense_and_sparse_history_have_the_same_events(self):
        event = {"_step": 0, "context_bias/step": 0, "context_bias/distance_s64": 2.5}
        keys = ["context_bias/step", "context_bias/distance_s64"]
        dense = [event, {"_step": 1, **dict.fromkeys(keys)}, {"_step": 2}]
        self.assertEqual(scan_metric_rows(History(dense), keys), [event])
        self.assertEqual(scan_metric_rows(History([event]), keys), [event])

    def test_partial_probe_is_rejected(self):
        keys = ["context_bias/step", "context_bias/distance_s64"]
        for row in ({keys[0]: 50, keys[1]: None}, {keys[0]: None, keys[1]: 1.2}):
            with self.subTest(row=row), self.assertRaisesRegex(RuntimeError, "Incomplete"):
                scan_metric_rows(History([row]), keys)


if __name__ == "__main__":
    unittest.main()
