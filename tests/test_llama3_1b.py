"""Check adapted 1B architecture, vocabulary, and portable example configurations."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchtitan.config import ConfigManager
from torchtitan.models.llama3 import llama3_configs

from torchtitan_dynamic_seq_len import config_registry as registry
from torchtitan_dynamic_seq_len.run_config import resolve_point

ROOT = Path(__file__).resolve().parents[1]


class Llama3OneBTest(unittest.TestCase):
    def test_adapted_architecture_parameter_count_and_tied_weights(self):
        config = registry.llama3_1b_llama2_adamw_constant_lr()
        native = llama3_configs["1B"](attn_backend="flex")
        with torch.device("meta"):
            model = config.model_spec.model.build()
            upstream = native.build()
        self.assertEqual(
            {
                name: tuple(p.shape)
                for name, p in model.named_parameters()
                if name.startswith("layers.")
            },
            {
                name: tuple(p.shape)
                for name, p in upstream.named_parameters()
                if name.startswith("layers.")
            },
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_038_682_112)
        self.assertIs(model.tok_embeddings.weight, model.lm_head.weight)
        self.assertEqual(tuple(model.tok_embeddings.weight.shape), (32000, 2048))
        self.assertEqual(len(model.layers), 16)
        self.assertEqual(native.vocab_size, 128256)
        self.assertEqual(native.tok_embeddings.num_embeddings, 128256)
        self.assertEqual(native.lm_head.out_features, 128256)
        self.assertEqual(config.loss.loss_fn.global_vocab_size, 32000)
        self.assertTrue(config.hf_assets_path.endswith("/hf/llama-tokenizer"))

    def test_every_trainer_uses_matching_model_tokenizer_and_loss(self):
        for suffix in ("", "_constant_lr", "_dynamic_seq", "_dynamic_seq_bs", "_seq_grad"):
            with self.subTest(suffix=suffix):
                config = getattr(registry, f"llama3_1b_llama2_adamw{suffix}")()
                self.assertEqual(config.model_spec.model.vocab_size, 32000)
                self.assertEqual(config.loss.loss_fn.global_vocab_size, 32000)
                self.assertTrue(config.hf_assets_path.endswith("/hf/llama-tokenizer"))
        old = registry.llama3_100m_llama2_adamw()
        self.assertEqual(old.model_spec.model.vocab_size, 32000)
        self.assertEqual(old.loss.loss_fn.global_vocab_size, 32000)
        self.assertTrue(old.hf_assets_path.endswith("/hf/llama-tokenizer"))

    def test_examples_parse_for_one_and_two_gpus(self):
        for filename in ("llama3_1b_smoke.yaml", "llama3_1b_fineweb.yaml"):
            for world in (1, 2):
                with (
                    self.subTest(filename=filename, world=world),
                    patch.dict(os.environ, {"WORLD_SIZE": str(world)}),
                ):
                    yaml, _, _, args = resolve_point(ROOT / "configs/examples" / filename, 0)
                    config = ConfigManager().parse_args(
                        [
                            "--module",
                            "torchtitan_dynamic_seq_len.config_registry",
                            "--config",
                            yaml["torchtitan_config"],
                            *args,
                        ]
                    )
                    self.assertEqual(config.training.steps, 10)
                    self.assertEqual(config.training.global_batch_size % world, 0)
                    self.assertEqual(config.training.local_batch_size, 1)
                    self.assertEqual(config.parallelism.data_parallel_replicate_degree, world)
                    self.assertEqual(config.loss.loss_fn.global_vocab_size, 32000)
                    self.assertFalse(config.checkpoint.last_save_model_only)
                    self.assertFalse(yaml["wandb"]["enabled"])
                    if "smoke" in filename:
                        self.assertTrue(Path(config.dataloader.dataset_path).is_file())
                        self.assertFalse(config.validator.enable)
                    else:
                        self.assertEqual(config.validator.eval_seq_len, 1024)
                        self.assertEqual(config.validator.token_budget, 20_971_520)

    def test_downloaded_tokenizer_if_available(self):
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        path = registry.llama3_1b_llama2_adamw().hf_assets_path
        if not (Path(path) / "tokenizer.json").is_file():
            self.skipTest("Prepare the Llama2 tokenizer first.")
        tokenizer = HuggingFaceTokenizer(tokenizer_path=path)
        self.assertEqual(tokenizer.vocab_size, 32000)
        tokens = tokenizer.encode("Hello world.", add_bos=True, add_eos=True)
        self.assertEqual((tokens[0], tokens[-1]), (1, 2))


if __name__ == "__main__":
    unittest.main()
