"""Check longer validation without changing short-context model behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.rope import RoPE

from torchtitan_dynamic_seq_len.config_registry import _model
from torchtitan_dynamic_seq_len.data.fixed_validation import FixedBudgetValidator
from torchtitan_dynamic_seq_len.seq_grad.trainer import SeqGradTrainer
from torchtitan_dynamic_seq_len.seq_grad.validation import SeqGradValidator


class SeparateValidationLengthTests(unittest.TestCase):
    def test_extended_cache_preserves_short_logits_and_gradients(self):
        torch.manual_seed(4)
        cfg = _model('flex', dim=64, num_layers=1, num_heads=4, num_kv_heads=2, vocab_size=128)
        for layer in cfg.layers:
            layer.attention.inner_attention = ScaledDotProductAttention.Config()
            layer.attention.rope.max_seq_len = 8
        model = cfg.build()
        model.init_states(buffer_device=torch.device('cpu'))
        tokens = torch.randint(0, 128, (1, 16))
        before = model(tokens[:, :8])
        before.square().sum().backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        caches = [m.cache.clone() for m in model.modules() if isinstance(m, RoPE)]
        trainer = object.__new__(SeqGradTrainer)
        trainer.model_parts = [model]
        trainer._ensure_rope_capacity(16)
        model.zero_grad()
        after = model(tokens[:, :8])
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        after.square().sum().backward()
        for old, parameter in zip(gradients, model.parameters(), strict=True):
            torch.testing.assert_close(old, parameter.grad, rtol=0, atol=0)
        for old, rope in zip(caches, [m for m in model.modules() if isinstance(m, RoPE)], strict=True):
            torch.testing.assert_close(old, rope.cache[:8], rtol=0, atol=0)
            self.assertEqual(rope.cache.shape[0], 16)
        self.assertEqual(tuple(model(tokens).shape), (1, 16, 128))

    def test_validator_receives_eval_capacity_and_checks_dataset_identity(self):
        cfg = SeqGradValidator.Config(eval_seq_len=24)
        cfg.dataloader = SimpleNamespace(dataset='fineweb_edu_long', expected_manifest_sha256='correct')

        def initialize(validator, config, **kwargs):
            self.assertEqual(kwargs['seq_len'], 24)
            validator.artifact = SimpleNamespace(manifest={'source': {'split': 'validation', 'manifest_sha256': 'correct'}})

        with patch.object(FixedBudgetValidator, '__init__', initialize):
            cfg.build(seq_len=16)
            cfg.dataloader.expected_manifest_sha256 = 'different'
            with self.assertRaisesRegex(ValueError, 'source differs'):
                cfg.build(seq_len=16)


if __name__ == '__main__':
    unittest.main()
