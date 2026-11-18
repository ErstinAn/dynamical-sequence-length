"""Check value comparisons independently of checkpoint serialization bytes."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from scripts.compare_checkpoints import compare


class CheckpointComparisonTest(unittest.TestCase):
    def test_tensor_and_counter_differences_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b, c = (Path(folder) / name for name in ("a", "b", "c"))
            state = {"model.weight": torch.arange(12).reshape(3, 4),
                     "train_state": {"step": 8, "tokens": 31457280}}
            dcp.save(state, checkpoint_id=a)
            dcp.save(state, checkpoint_id=b)
            self.assertTrue(compare(a, b)["exact_match"])
            state["model.weight"][0, 0] = 1
            state["train_state"]["tokens"] -= 1
            dcp.save(state, checkpoint_id=c)
            result = compare(a, c)
            self.assertFalse(result["exact_match"])
            self.assertEqual({row["key"] for row in result["differences"]},
                             {"model.weight", "train_state.tokens"})


if __name__ == "__main__":
    unittest.main()
