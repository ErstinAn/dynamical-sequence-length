"""Check independent evaluation lengths and packed-document safeguards."""

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torchtitan.config import ParallelismConfig
from torchtitan.models.common.attention import ScaledDotProductAttention

from tests.test_fixed_validation import _dims, _loss, _Model, _write_artifact
from torchtitan_dynamic_seq_len.config_registry import llama3_100m_long_documents_x_monitor
from torchtitan_dynamic_seq_len.data.packed_documents import PACKING_PROTOCOL
from torchtitan_dynamic_seq_len.grad_var.probes import validate_config
from torchtitan_dynamic_seq_len.grad_var.validation import XValidator, required_context_length


class XValidationTests(unittest.TestCase):
    def test_longer_validation_uses_all_tokens_and_restores_model_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_artifact(root, n=12, length=16)
            validator = XValidator.Config(
                token_budget=manifest["token_budget"],
                artifact_path=str(root),
                eval_seq_len=16,
                eval_local_batch_size=5,
            ).build(
                parallelism=ParallelismConfig(),
                dp_world_size=1,
                dp_rank=0,
                tokenizer=SimpleNamespace(tokenizer_path=str(root / "tokenizer")),
                parallel_dims=_dims(),
                loss_fn=_loss,
                validation_context=nullcontext,
                metrics_processor=SimpleNamespace(ntokens_since_last_log=0, log_validation=Mock()),
                seq_len=8,
                local_batch_size=32,
            )
            model = _Model()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            validator.validate([model], 1)
            self.assertEqual(validator.last_result["evaluated_tokens"], 192)
            self.assertEqual(validator.last_result["forwards_per_rank"], 3)
            self.assertTrue(all(batch.shape[1] == 16 for batch in model.seen))
            self.assertTrue(model.training)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_context_limit_includes_validation_and_packed_data_requires_masks(self):
        config = llama3_100m_long_documents_x_monitor()
        config.training.seq_len = 4096
        config.validator.enable = True
        config.validator.eval_seq_len = 22256
        config.validator.token_budget = 20830722
        self.assertIsInstance(config.validator, XValidator.Config)
        with tempfile.TemporaryDirectory() as directory:
            config.validator.artifact_path = directory
            manifest = {
                "schema_version": 2,
                "sequence_length": 22256,
                "token_budget": 20830722,
                "packing_protocol": PACKING_PROTOCOL,
                "source": {"split": "validation"},
            }
            (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
            validate_config(config)
            self.assertEqual(required_context_length(config), 22256)
            self.assertEqual(config.training.seq_len, 4096)
            attention = config.model_spec.model.layers[0].attention
            original = attention.inner_attention
            attention.inner_attention = ScaledDotProductAttention.Config()
            with self.assertRaisesRegex(ValueError, "document-masked"):
                validate_config(config)
            attention.inner_attention = original
            config.validator.eval_seq_len = 50000
            with self.assertRaisesRegex(ValueError, "position limit"):
                validate_config(config)
            config.validator.enable = False
            self.assertEqual(required_context_length(config), 4096)


if __name__ == "__main__":
    unittest.main()
