"""Regression: loading a reusable test baseline must not alias its state."""

import copy
import unittest
from types import SimpleNamespace

import torch
from torchtitan.components.lr_scheduler import LRSchedulersContainer

from tests.seq_grad_microbatch_gpu import assert_identical, capture_fixture_state, restore_fixture_state
from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers


class MutableLoader:
    def __init__(self):
        self.packing_buffer = []

    def state_dict(self):
        return {"packing_buffer": self.packing_buffer}

    def load_state_dict(self, state):
        self.packing_buffer = state["packing_buffer"]


class FixtureIsolationTest(unittest.TestCase):
    def test_baseline_survives_two_restores_and_training_trajectories(self):
        model = torch.nn.Linear(3, 3)
        optimizers = ExperimentOptimizers(ExperimentOptimizers.Config(), model_parts=[model])
        trainer = SimpleNamespace(model_parts=[model], optimizers=optimizers,
            lr_schedulers=LRSchedulersContainer.Config(warmup_steps=0, min_lr_factor=1.).build(
                optimizers=optimizers, training_steps=4),
            dataloader=MutableLoader(), device=torch.device("cpu"))
        baseline = capture_fixture_state(trainer)
        original = copy.deepcopy(baseline)
        results = []
        for _ in range(2):
            restore_fixture_state(trainer, baseline)
            for step in range(4):
                optimizers.zero_grad()
                model(torch.ones(2, 3)).square().mean().backward()
                optimizers.step()
                trainer.lr_schedulers.step()
                trainer.dataloader.packing_buffer.append(step)
            results.append(capture_fixture_state(trainer))
            assert_identical(baseline, original)
        assert_identical(*results)


if __name__ == "__main__":
    unittest.main()
