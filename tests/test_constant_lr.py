"""Do checks of the TorchTitan sweep and the learning rates for AdamW updates."""

import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import ConfigManager
from torchtitan.trainer import Trainer

from torchtitan_dynamic_seq_len.config_registry import llama3_100m_llama2_adamw_constant_lr
from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers
from torchtitan_dynamic_seq_len.run_config import grid, load_config, resolve_point

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "configs/lr_sweep_constant.yaml"


class ConstantLearningRateTest(unittest.TestCase):
    def test_every_sweep_point_builds_native_trainer_with_fixed_batch_and_length(self):
        keys, points = grid(load_config(SWEEP))
        self.assertEqual(keys, ["optimizer.learning_rate"])
        self.assertEqual([point[0] for point in points], [0.0005, 0.001, 0.002, 0.004, 0.008])
        for index, (learning_rate,) in enumerate(points):
            raw, _, _, args = resolve_point(SWEEP, index)
            env = {
                "WORLD_SIZE": "2",
                "DYNAMIC_SEQ_LEN_VALIDATION_DIR": raw["paths"]["validation_dir"],
            }
            with self.subTest(learning_rate=learning_rate), patch.dict(os.environ, env):
                config = ConfigManager().parse_args(
                    [
                        "--module",
                        "torchtitan_dynamic_seq_len.config_registry",
                        "--config",
                        raw["torchtitan_config"],
                        *args,
                    ]
                )
                # Do a check of Config.build without GPU process initialization.
                with patch.object(Trainer, "__init__", return_value=None):
                    self.assertIs(type(config.build()), Trainer)
                self.assertIs(type(config.metrics), MetricsProcessor.Config)
                self.assertFalse(hasattr(config, "grad_var_monitor"))
                self.assertEqual(config.training.global_batch_size, 512)
                self.assertEqual(config.training.local_batch_size, 16)
                self.assertEqual(config.training.seq_len, 1024)
                self.assertEqual(config.training.steps, 3819)
                self.assertEqual(config.optimizer.learning_rate, learning_rate)
                self.assertEqual(config.dataloader.num_workers, 0)
                self.assertIsNone(config.dataloader.prefetch_factor)
                self.assertFalse(config.dataloader.persistent_workers)
                self.assertEqual(config.lr_scheduler.warmup_steps, 0)
                self.assertEqual(config.lr_scheduler.min_lr_factor, 1.0)

    def test_constant_updates_and_checkpoint_resume(self):
        config = llama3_100m_llama2_adamw_constant_lr()

        def build():
            model = torch.nn.Linear(3, 2)
            optimizers = ExperimentOptimizers(config.optimizer, model_parts=[model])
            scheduler = config.lr_scheduler.build(optimizers=optimizers, training_steps=8)
            return model, optimizers, scheduler

        model, optimizers, scheduler = build()
        resumed = None
        # Include the first update, the last planned update, and the updates after it.
        for step in range(1, 13):
            for current in [(model, optimizers, scheduler), *([resumed] if resumed else [])]:
                net, opts, sched = current
                self.assertEqual(len(opts.optimizers), 2)
                for optimizer in opts.optimizers:
                    for group in optimizer.param_groups:
                        self.assertEqual(group["lr"], config.optimizer.learning_rate)
                for param in net.parameters():
                    param.grad = torch.ones_like(param)
                opts.step()
                sched.step()
                self.assertTrue(
                    all(lr == config.optimizer.learning_rate for lr in sched.get_metrics().values())
                )
            if step == 4:
                resumed = build()
                resumed[0].load_state_dict(copy.deepcopy(model.state_dict()))
                for target, source in zip(resumed[1].optimizers, optimizers.optimizers):
                    target.load_state_dict(copy.deepcopy(source.state_dict()))
                resumed[2].load_state_dict(copy.deepcopy(scheduler.state_dict()))
            if resumed:
                self.assertEqual(resumed[2].state_dict(), scheduler.state_dict())
                for actual, expected in zip(resumed[0].parameters(), model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
