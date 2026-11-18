"""Check the main launch and recovery commands without starting training."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import train


def configuration(root):
    # Test values only. These are not the experiment's proposed hyperparameters.
    return {
        "name": "launch-test", "torchtitan_config": "llama3_1b",
        "paths": {"dump_root": str(root / "runs")},
        "wandb": {"enabled": True, "mode": "online", "entity": "test", "project": "test"},
        "fixed": {
            "token_budget": 1280, "training.steps": 20, "training.seq_len": 16384,
            "training.local_batch_size": 1, "training.global_batch_size": 2,
            "seq_len_scheduler.kind": "constant", "seq_len_scheduler.length": 32,
            "batch_size_scheduler.kind": "constant", "batch_size_scheduler.batch_size": 2,
            "microbatch_scheduler.kind": "constant", "microbatch_scheduler.batch_size": 1,
            "optimizer.learning_rate": 0.001, "lr_scheduler.warmup_steps": 0,
            "lr_scheduler.decay_type": "linear", "lr_scheduler.min_lr_factor": 1.0,
            "validator.freq": 2,
        },
    }


class MainLaunchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "test.yaml"
        self.config = configuration(self.root)
        self.write_config()

    def write_config(self):
        self.path.write_text(yaml.safe_dump(self.config))

    def invoke(self, args):
        output = io.StringIO()
        with patch.object(sys, "argv", ["train.py", *map(str, args)]), contextlib.redirect_stdout(output):
            train.main()
        return output.getvalue()

    def test_plan_is_concrete_and_dry_run_has_no_files(self):
        output = self.invoke([self.path, "--nproc", "2", "--dry-run"])
        self.assertIn('"actual_tokens": 1280', output)
        self.assertIn('"index": 10', output)
        self.assertIn("--module torchtitan_dynamic_seq_len.experiment", output)
        self.assertFalse((self.root / "runs").exists())

    def test_rejects_missing_settings_and_disabled_wandb(self):
        for change in ("wandb", "budget", "validation"):
            self.config = configuration(self.root)
            if change == "wandb":
                self.config["wandb"]["enabled"] = False
            elif change == "budget":
                del self.config["fixed"]["token_budget"]
            else:
                self.config["fixed"]["validator.eval_seq_len"] = 22256
            self.write_config()
            with self.subTest(change=change), self.assertRaises(SystemExit) as error:
                self.invoke([self.path, "--dry-run"])
            self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "runs").exists())

    def test_rejects_microbatch_that_only_fits_the_first_phase(self):
        self.config["fixed"].update({
            "microbatch_scheduler.kind": "increasing",
            "microbatch_scheduler.batch_sizes": [1, 2],
            "microbatch_scheduler.boundaries": [2],
        })
        self.write_config()
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            self.invoke([self.path, "--nproc", "2", "--dry-run"])
        self.assertIn("Invalid schedules at step 3", errors.getvalue())
        self.assertFalse((self.root / "runs").exists())

    def test_agreed_smoke_and_controlled_stop_parse_without_launch(self):
        path = train.ROOT / "configs/smoke.yaml"
        output = self.invoke([path, "--nproc", "4", "--stop-after-checkpoint", "4", "--dry-run"])
        self.assertIn('"actual_tokens": 31457280', output)
        self.assertIn("--smoke_test", output)

    def test_resume_preserves_directory_and_uses_new_wandb_segment(self):
        with patch("torch.cuda.device_count", return_value=1), patch.object(train.subprocess, "call", return_value=0) as launch:
            with self.assertRaises(SystemExit) as error:
                self.invoke([self.path])
            self.assertEqual(error.exception.code, 0)
            first_env = launch.call_args.kwargs["env"]
            dump = next((self.root / "runs/launch-test").iterdir())
            checkpoint = dump / "checkpoint/step-2"
            checkpoint.mkdir(parents=True)
            (checkpoint / ".metadata").touch()
            original = (dump / "launch.json").read_bytes()
            with self.assertRaises(SystemExit) as error:
                self.invoke(["--resume-dir", dump])
            self.assertEqual(error.exception.code, 0)
            second_env = launch.call_args.kwargs["env"]
            self.assertNotEqual(first_env["WANDB_RUN_ID"], second_env["WANDB_RUN_ID"])
            self.assertEqual(first_env["WANDB_RUN_GROUP"], second_env["WANDB_RUN_GROUP"])
            self.assertEqual((dump / "launch.json").read_bytes(), original)
            segments = [json.loads(p.read_text()) for p in dump.glob("segment-*.json")]
            self.assertEqual(sorted(s["resume_step"] for s in segments), [0, 2])
            self.assertEqual(len(list((self.root / "runs/launch-test").iterdir())), 1)
            with self.assertRaises(SystemExit) as error:
                self.invoke(["--resume-dir", dump, "--nproc", "2", "--dry-run"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
