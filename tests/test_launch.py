"""Check the portable launch settings without a GPU."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from torchtitan.config import ConfigManager

from torchtitan_dynamic_seq_len.run_config import resolve_point

ROOT = Path(__file__).resolve().parents[1]


class LocalLaunchTest(unittest.TestCase):
    def test_smoke_config_uses_local_data_and_ten_steps(self):
        config, _, _, args = resolve_point(ROOT / "configs/smoke.yaml", 0)
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            parsed = ConfigManager().parse_args(
                [
                    "--module",
                    "torchtitan_dynamic_seq_len.config_registry",
                    "--config",
                    config["torchtitan_config"],
                    *args,
                ]
            )
        self.assertEqual(parsed.training.steps, 10)
        self.assertEqual(parsed.training.seq_len, 64)
        self.assertEqual(parsed.dataloader.num_workers, 0)
        self.assertIsNone(parsed.dataloader.prefetch_factor)
        self.assertTrue(Path(parsed.dataloader.dataset_path).is_file())
        self.assertTrue(parsed.debug.deterministic)
        self.assertTrue(parsed.grad_var_monitor.enable)

    def test_dry_run_does_not_create_output(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            env = {**os.environ, "DYNAMIC_SEQ_LEN_OUTPUT": str(output)}
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/train.py"),
                    str(ROOT / "configs/seq512.yaml"),
                    "--nproc",
                    "2",
                    "--dry-run",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("--nproc_per_node=2", result.stdout)
            self.assertIn("--training.seq_len 512", result.stdout)
            self.assertFalse(output.exists())

    def test_asset_root_changes_both_data_and_tokenizer_paths(self):
        with patch.dict(os.environ, {"DYNAMIC_SEQ_LEN_ASSETS": "/tmp/selected-assets"}):
            config, _, _, _ = resolve_point(ROOT / "configs/seq1024.yaml", 0)
        self.assertEqual(
            config["fixed"]["hf_assets_path"], "/tmp/selected-assets/hf/llama-tokenizer"
        )
        self.assertEqual(
            config["paths"]["validation_dir"],
            "/tmp/selected-assets/data/fineweb100_validation_llama2_1024",
        )

    def test_unknown_optimizer_method_is_rejected(self):
        import torch

        from torchtitan_dynamic_seq_len.optimizer import ExperimentOptimizers

        with self.assertRaisesRegex(ValueError, "optimizer.method"):
            ExperimentOptimizers(
                ExperimentOptimizers.Config(method="soap"), model_parts=[torch.nn.Linear(4, 4)]
            )


if __name__ == "__main__":
    unittest.main()
