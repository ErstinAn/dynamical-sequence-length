from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from run_from_yaml import (
    config_digest,
    latest_complete_step,
    load_manifest,
    manifest_entry,
    wandb_entity,
    writer_lock,
)
from storage_paths import storage_path
from submit import (
    atomic_write_submission,
    make_manifest,
    submission_environment,
    validate_resume_runs,
)

from torchtitan_dynamic_seq_len.run_config import load_config, source_digest

HERE = Path(__file__).resolve().parents[1] / "cluster/vt_arc"
SOURCE_CONFIG = HERE.parents[1] / "configs/seq512.yaml"


class LauncherTest(unittest.TestCase):
    def test_submission_does_not_export_login_node_wandb_service(self):
        with patch.dict(
            os.environ, {"WANDB_SERVICE": "login-node-socket", "WANDB_PROJECT": "debug"}
        ):
            environment = submission_environment(Path("config.yaml"), Path("manifest.yaml"))
            self.assertNotIn("WANDB_SERVICE", environment)
            self.assertEqual(environment["WANDB_PROJECT"], "debug")
            self.assertEqual(os.environ["WANDB_SERVICE"], "login-node-socket")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config = load_config(SOURCE_CONFIG)
        config["resources"] = {
            "account": "test_account",
            "cluster": "tinkercliffs",
            "partition": "h200_normal_q",
            "qos": "tc_h200_normal_short",
            "nodes": 1,
            "gpus_per_node": 2,
            "cpus_per_task": 16,
            "time": "02:00:00",
            "max_concurrent": 1,
        }
        # Generic launcher tests use a single offline point; the formal three-length
        # experiment is checked separately in test_formal_experiment.py.
        config["fixed"]["training.local_batch_size"] = 16
        config["sweep"] = {}
        config["wandb"]["enabled"] = False
        config["paths"] = {
            "dump_root": str(self.root / "runs"),
            "slurm_log_root": str(self.root / "logs"),
            "manifest_root": str(self.root / "manifests"),
            "hf_home": str(self.root / "hf"),
            "torch_home": str(self.root / "torch"),
        }
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_entry_points_without_pythonpath(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        for script in ("submit.py", "run_from_yaml.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(HERE / script), "--help"],
                    cwd=self.root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_wandb_entity_and_legacy_team_alias(self) -> None:
        self.assertEqual(wandb_entity({"entity": "dynamic_seq_length"}), "dynamic_seq_length")
        self.assertEqual(wandb_entity({"team": "legacy"}), "legacy")
        with self.assertRaisesRegex(ValueError, "disagree"):
            wandb_entity({"entity": "one", "team": "two"})

    def test_new_submissions_get_distinct_ids_and_directories(self) -> None:
        first = make_manifest(self.config_path, "new", ["a" * 32])
        second = make_manifest(self.config_path, "new", ["b" * 32])
        self.assertNotEqual(first["runs"][0]["wandb_id"], second["runs"][0]["wandb_id"])
        self.assertNotEqual(first["runs"][0]["dump_folder"], second["runs"][0]["dump_folder"])

    def test_resume_requires_complete_dcp_metadata(self) -> None:
        manifest = make_manifest(self.config_path, "resume", ["resume123"])
        with self.assertRaisesRegex(ValueError, "no complete DCP checkpoint"):
            validate_resume_runs(self.config, manifest)

        checkpoint = Path(manifest["runs"][0]["dump_folder"]) / "checkpoint"
        (checkpoint / "step-10").mkdir(parents=True)
        (checkpoint / "step-10" / ".metadata").touch()
        (checkpoint / "step-20").mkdir()
        run_record = {
            "wandb_id": "resume123",
            "config_sha256": config_digest(self.config),
            "source_sha256": source_digest(),
        }
        (checkpoint.parent / "run_manifest.yaml").write_text(
            yaml.safe_dump(run_record), encoding="utf-8"
        )
        validate_resume_runs(self.config, manifest)
        self.assertEqual(manifest["runs"][0]["resume_step"], 10)
        self.assertEqual(latest_complete_step(checkpoint), 10)

    def test_manifest_round_trip_binds_id_to_grid_point(self) -> None:
        manifest = make_manifest(self.config_path, "new", ["run123"])
        path = self.root / "manifest.yaml"
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        loaded = load_manifest(path)
        entry = manifest_entry(self.config_path, loaded, 0)
        self.assertEqual(entry["wandb_id"], "run123")
        self.assertEqual(loaded["config_sha256"], config_digest(self.config))

    def test_writer_lock_rejects_a_second_writer_and_cleans_up(self) -> None:
        dump = self.root / "run"
        dump.mkdir()
        entry = {"wandb_id": "run123"}
        with writer_lock(dump, entry):
            with self.assertRaisesRegex(RuntimeError, "writer lock"):
                with writer_lock(dump, entry):
                    pass
        self.assertFalse((dump / ".writer-lock").exists())

    def test_submission_persists_an_immutable_config_snapshot(self) -> None:
        manifest = make_manifest(self.config_path, "new", ["run123"])
        manifest_path, config_path = atomic_write_submission(
            self.root / "manifests", manifest, self.config
        )
        self.assertTrue(manifest_path.is_file())
        self.assertTrue(config_path.is_file())
        loaded = load_manifest(manifest_path)
        self.assertEqual(loaded["config_snapshot"], str(config_path))
        self.assertEqual(
            loaded["config_sha256"],
            config_digest(yaml.safe_load(config_path.read_text(encoding="utf-8"))),
        )

    def test_submission_paths_are_in_process_environment(self) -> None:
        config_path = self.root / "snapshot" / "config.yaml"
        manifest_path = self.root / "snapshot" / "manifest.yaml"
        environment = submission_environment(config_path, manifest_path)
        self.assertEqual(environment["SLURM_LAUNCHER_DIR"], str(HERE))
        self.assertEqual(environment["RUN_CONFIG"], str(config_path))
        self.assertEqual(environment["RUN_MANIFEST"], str(manifest_path))

    def test_submit_dry_run_uses_plain_export_all(self) -> None:
        result = subprocess.run(
            [sys.executable, str(HERE / "submit.py"), str(self.config_path), "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--export=ALL ", result.stdout)
        self.assertNotIn("--export=ALL,", result.stdout)
        self.assertIn("Submission environment:", result.stdout)

    def test_resume_one_sweep_index_with_walltime_override(self) -> None:
        self.config["sweep"]["optimizer.learning_rate"] = [0.001, 0.002]
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        previous = make_manifest(self.config_path, "new", ["first", "second"])
        previous_path = self.root / "previous.yaml"
        previous_path.write_text(yaml.safe_dump(previous), encoding="utf-8")

        selected_dump = Path(previous["runs"][1]["dump_folder"])
        checkpoint = selected_dump / "checkpoint" / "step-20"
        checkpoint.mkdir(parents=True)
        (checkpoint / ".metadata").touch()
        (selected_dump / "run_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "wandb_id": "second",
                    "config_sha256": config_digest(self.config),
                    "source_sha256": source_digest(),
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "submit.py"),
                str(self.config_path),
                "--resume-manifest",
                str(previous_path),
                "--resume-index",
                "1",
                "--time",
                "15:00:00",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode: resume", result.stdout)
        self.assertIn("resume_step: 20", result.stdout)
        self.assertIn("--time=15:00:00", result.stdout)
        self.assertIn("--array=1%1", result.stdout)
        self.assertNotIn("wandb_id: first", result.stdout)

    def test_compute_side_dry_run_uses_manifest_id_and_directory(self) -> None:
        manifest = make_manifest(self.config_path, "new", ["run123"])
        manifest_path, config_path = atomic_write_submission(
            self.root / "manifests", manifest, self.config
        )
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "run_from_yaml.py"),
                str(config_path),
                "--manifest",
                str(manifest_path),
                "--index",
                "0",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=new", result.stdout)
        self.assertIn("wandb_id=run123", result.stdout)
        self.assertIn(f"/{self.config['name']}/run123", result.stdout)
        self.assertIn("--metrics.no-enable_wandb", result.stdout)

    def test_launch_rejects_changed_source(self):
        from unittest.mock import patch

        from run_from_yaml import main as compute_main

        manifest = make_manifest(self.config_path, "new", ["source123"])
        manifest["source_sha256"] = "changed"
        manifest_path, snapshot = atomic_write_submission(
            self.root / "manifests", manifest, self.config
        )
        with patch.object(
            sys,
            "argv",
            ["run_from_yaml.py", str(snapshot), "--manifest", str(manifest_path), "--dry-run"],
        ):
            with self.assertRaisesRegex(ValueError, "source files changed"):
                compute_main()

    def test_storage_paths_do_not_change_the_selected_directory(self):
        self.assertEqual(storage_path("/tmp/example/run"), Path("/tmp/example/run"))

    def test_resume_rejects_changed_config_and_run_record(self):
        manifest = make_manifest(self.config_path, "resume", ["resume123"])
        dump = Path(manifest["runs"][0]["dump_folder"])
        step = dump / "checkpoint/step-20"
        step.mkdir(parents=True)
        (step / ".metadata").touch()
        record = dump / "run_manifest.yaml"
        record.write_text(
            yaml.safe_dump(
                {
                    "wandb_id": "resume123",
                    "config_sha256": config_digest(self.config),
                    "source_sha256": source_digest(),
                }
            )
        )
        validate_resume_runs(self.config, manifest)
        self.assertEqual(manifest["runs"][0]["resume_step"], 20)
        with patch("submit.source_digest", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "source does not match"):
                validate_resume_runs(self.config, manifest)
        changed = dict(self.config)
        changed["name"] = "changed"
        self.config_path.write_text(yaml.safe_dump(changed))
        with self.assertRaisesRegex(ValueError, "does not match"):
            manifest_entry(self.config_path, manifest, 0)
        record.write_text(yaml.safe_dump({"wandb_id": "resume123", "config_sha256": "wrong"}))
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_resume_runs(self.config, manifest)

    def test_cluster_selection_and_missing_account(self) -> None:
        self.config["resources"]["cluster"] = "owl"
        self.config["resources"]["partition"] = "b200_normal_q"
        self.config["resources"]["qos"] = "owl_b200_normal_short"
        self.config_path.write_text(yaml.safe_dump(self.config))
        cmd = [sys.executable, str(HERE / "submit.py"), str(self.config_path), "--dry-run"]
        result = subprocess.run(cmd, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--partition=b200_normal_q", result.stdout)
        self.assertIn("--clusters=owl", result.stdout)
        self.config["resources"]["account"] = ""
        self.config_path.write_text(yaml.safe_dump(self.config))
        result = subprocess.run(cmd, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set resources.account", result.stderr)


if __name__ == "__main__":
    unittest.main()
