"""Validate the actual 100M/Llama2 YAML, tokenizer, and frozen validation artifact."""

import copy
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from run_from_yaml import main as compute_main
from run_from_yaml import resolve_point
from submit import atomic_write_submission, make_manifest
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ConfigManager

from torchtitan_dynamic_seq_len.config_registry import llama3_100m_llama2_adamw
from torchtitan_dynamic_seq_len.data.fineweb100 import (
    load_evaluation_exclusions,
    validation_directory,
)
from torchtitan_dynamic_seq_len.data.fineweb100_identity import verify_validation_identity
from torchtitan_dynamic_seq_len.data.fixed_validation import (
    FrozenValidationArtifact,
    resolve_artifact_path,
)
from torchtitan_dynamic_seq_len.grad_var_trainer import GradVarTrainer
from torchtitan_dynamic_seq_len.run_config import load_config

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "configs/seq512.yaml"
YAML_PATHS = [YAML_PATH, *(YAML_PATH.with_name(f"seq{length}.yaml") for length in (1024, 2048))]


class FormalExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.yaml = load_config(YAML_PATH)
        cls.config = llama3_100m_llama2_adamw()
        with torch.device("meta"):
            cls.model = cls.config.model_spec.model.build()
        cls.parameter_count = sum(p.numel() for p in cls.model.parameters() if p.requires_grad)

    def test_real_model_count_tied_vocab_and_rounded_20tpp(self):
        self.assertEqual(self.parameter_count, 100_092_672)
        self.assertIs(self.model.tok_embeddings.weight, self.model.lm_head.weight)
        if not Path(self.config.hf_assets_path, "tokenizer.json").is_file():
            self.skipTest("Run scripts/prepare_tokenizer.py to check the tokenizer.")
        tokenizer = HuggingFaceTokenizer(tokenizer_path=self.config.hf_assets_path)
        self.assertEqual(tokenizer.vocab_size, 32000)
        self.assertEqual(self.config.model_spec.model.vocab_size, tokenizer.vocab_size)
        self.assertEqual(self.config.loss.loss_fn.global_vocab_size, tokenizer.vocab_size)
        self.assertEqual(self.model.tok_embeddings.weight.shape, (32000, 768))
        self.assertEqual(len(self.model.layers), 12)
        tokens = tokenizer.encode("Hello world. 中文 tokenizer check.", add_bos=True, add_eos=True)
        self.assertEqual((tokens[0], tokens[-1]), (1, 2))
        self.assertTrue(all(0 <= token < 32000 for token in tokens))
        self.assertTrue(Path(tokenizer.tokenizer_path).is_relative_to(ROOT))
        for path in YAML_PATHS:
            fixed = load_config(path)["fixed"]
            per_step = fixed["training.global_batch_size"] * fixed["training.seq_len"]
            expected_steps = math.ceil(20 * self.parameter_count / per_step)
            self.assertEqual(fixed["training.steps"], expected_steps)
            self.assertEqual(fixed["lr_scheduler.warmup_steps"], expected_steps // 10)
            self.assertLess((expected_steps - 1) * per_step, 20 * self.parameter_count)
            self.assertGreaterEqual(expected_steps * per_step, 20 * self.parameter_count)

    def test_three_lengths_parse_into_matching_trainer_and_dp(self):
        names = set()
        for path, length, steps in zip(YAML_PATHS, (512, 1024, 2048), (7637, 3819, 1910)):
            config, _, parameters, args = resolve_point(path, 0)
            names.add(config["name"])
            env = {
                "WORLD_SIZE": "2",
                "DYNAMIC_SEQ_LEN_VALIDATION_DIR": config["paths"]["validation_dir"],
            }
            with self.subTest(sequence_length=length), patch.dict(os.environ, env):
                parsed = ConfigManager().parse_args(
                    [
                        "--module",
                        "torchtitan_dynamic_seq_len.config_registry",
                        "--config",
                        config["torchtitan_config"],
                        *args,
                        "--metrics.enable_wandb",
                    ]
                )
                self.assertIsInstance(parsed, GradVarTrainer.Config)
                self.assertTrue(parsed.grad_var_monitor.enable)
                self.assertTrue(parsed.metrics.enable_wandb)
                self.assertEqual(parsed.training.local_batch_size, 16)
                self.assertEqual(parameters, {})
                self.assertEqual(parsed.training.global_batch_size, 512)
                self.assertEqual(parsed.training.seq_len, length)
                self.assertEqual(parsed.training.steps, steps)
                self.assertEqual(parsed.parallelism.data_parallel_replicate_degree, 2)
                self.assertEqual(parsed.parallelism.data_parallel_shard_degree, 1)
                self.assertEqual(parsed.training.global_batch_size // (16 * 2), 16)
                self.assertEqual(
                    resolve_artifact_path(parsed.validator.artifact_path),
                    validation_directory().resolve(),
                )
                self.assertEqual(parsed.hf_assets_path, self.config.hf_assets_path)
        self.assertEqual(len(names), 3)

    @unittest.skipUnless(
        os.environ.get("DYNAMIC_SEQ_LEN_ARTIFACT_TEST") == "1",
        "Set DYNAMIC_SEQ_LEN_ARTIFACT_TEST=1 to check local data.",
    )
    def test_built_artifact_checksums_vocabulary_and_training_exclusions(self):
        import numpy as np

        reference = FrozenValidationArtifact(
            ROOT / "assets/data/fineweb100_validation_llama2_1024",
            token_budget=20_971_520,
            sequence_length=1024,
            tokenizer_path=self.config.hf_assets_path,
        )
        for length in (512, 1024, 2048, 4096):
            artifact = FrozenValidationArtifact(
                ROOT / f"assets/data/fineweb100_validation_llama2_{length}",
                token_budget=20_971_520,
                sequence_length=length,
                tokenizer_path=self.config.hf_assets_path,
            )
            self.assertEqual(artifact.num_sequences, 20_971_520 // length)
            for name in ("input", "labels"):
                self.assertLess(int(artifact.arrays[name].max()), 32000)
                # Length changes only context grouping: evaluate the same ordered token stream.
                np.testing.assert_array_equal(
                    artifact.arrays[name].reshape(-1), reference.arrays[name].reshape(-1)
                )
            self.assertLess(int(artifact.arrays["positions"].max()), length)
            self.assertEqual(
                artifact.manifest["evaluation_documents"],
                reference.manifest["evaluation_documents"],
            )
            ids, hashes = load_evaluation_exclusions(artifact.path)
            self.assertEqual(len(ids), artifact.manifest["evaluation_documents"]["count"])
            self.assertTrue(hashes)
        old = json.loads(
            (
                ROOT
                / "torchtitan_dynamic_seq_len/data/manifests/fineweb_100bt_validation_reference.json"
            ).read_text()
        )
        old = old["artifact_identity"]
        verify_validation_identity(old)
        mixed = copy.deepcopy(artifact.manifest)
        mixed["arrays"] = old["arrays"]
        with self.assertRaisesRegex(ValueError, "versioned reference"):
            verify_validation_identity(mixed)
        with self.assertRaisesRegex(ValueError, "sequence length"):
            FrozenValidationArtifact(
                artifact.path,
                token_budget=20_971_520,
                sequence_length=512,
                tokenizer_path=self.config.hf_assets_path,
            )

    def test_launcher_exports_exact_wandb_target_and_validation_dir(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            config = copy.deepcopy(self.yaml)
            config["resources"] = {"gpus_per_node": 2}
            config["wandb"] = {
                "enabled": True,
                "entity": "test_entity",
                "project": "test_project",
                "mode": "offline",
            }
            for key in ("dump_root", "slurm_log_root", "manifest_root", "hf_home", "torch_home"):
                config["paths"][key] = str(Path(tmp) / key)
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            manifest = make_manifest(config_path, "new", ["testvar0"])
            self.assertEqual(len(manifest["runs"]), 1)
            self.assertEqual(len({r["run_name"] for r in manifest["runs"]}), 1)
            manifest_path, snapshot = atomic_write_submission(
                Path(tmp) / "manifests", manifest, config
            )
            os.environ.pop("WANDB_MODE", None)
            with (
                patch(
                    "sys.argv",
                    [
                        "run_from_yaml.py",
                        str(snapshot),
                        "--manifest",
                        str(manifest_path),
                        "--index",
                        "0",
                        "--dry-run",
                    ],
                ),
                redirect_stdout(StringIO()) as output,
            ):
                compute_main()
            self.assertEqual(os.environ["WANDB_TEAM"], "test_entity")
            self.assertEqual(os.environ["WANDB_ENTITY"], "test_entity")
            self.assertEqual(os.environ["WANDB_PROJECT"], "test_project")
            self.assertEqual(os.environ["WANDB_MODE"], "offline")
            self.assertEqual(
                os.environ["DYNAMIC_SEQ_LEN_VALIDATION_DIR"], config["paths"]["validation_dir"]
            )
            self.assertIn("--metrics.enable_wandb", output.getvalue())
            self.assertIn("--nproc 2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
