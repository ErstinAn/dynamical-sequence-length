"""Check compile scope, parameter identity, and W&B failure reporting."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torchtitan.components.metrics import LoggerContainer, WandBLogger
from torchtitan.config import CompileConfig

from torchtitan_dynamic_seq_len.dynamic_seq.metrics import DynamicSeqMetricsProcessor
from torchtitan_dynamic_seq_len.dynamic_seq_bs.compile import compile_dynamic_blocks


class DynamicCompileTest(unittest.TestCase):
    def setUp(self):
        previous = torch.compiler.config.static_sources
        self.addCleanup(setattr, torch.compiler.config, "static_sources", previous)
        self.addCleanup(torch._dynamo.reset)
        torch.compiler.config.static_sources = "existing_source"

    def test_dynamic_wrapper_is_lazy_and_preserves_parameters(self):
        block = torch.nn.Linear(3, 2)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict({"0": block})
        original = list(model.parameters())
        with patch("torch.compile", wraps=torch.compile) as compile_call:
            compile_dynamic_blocks([model], CompileConfig(enable=True, backend="eager"))
            self.assertEqual(
                compile_call.call_args.kwargs,
                {
                    "backend": "eager",
                    "fullgraph": True,
                    "dynamic": True,
                },
            )
        self.assertTrue(all(a is b for a, b in zip(original, model.parameters())))
        self.assertEqual(torch.compiler.config.static_sources, r"existing_source,.*\.head_dim")
        for batch in (2, 3, 5, 8, 13, 21, 34, 55, 57, 96):
            x = torch.randn(batch, 3)
            expected = torch.nn.functional.linear(x, block.weight, block.bias)
            torch.testing.assert_close(block(x), expected)

    def test_disabled_or_loss_only_compile_keeps_blocks_unchanged(self):
        block = Mock()
        model = SimpleNamespace(layers={"0": block})
        compile_dynamic_blocks([model], CompileConfig(enable=False))
        compile_dynamic_blocks([model], CompileConfig(enable=True, components=["loss"]))
        block.compile.assert_not_called()
        self.assertEqual(torch.compiler.config.static_sources, "existing_source")

    def test_wandb_failure_and_normal_close(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                client = Mock()
                client.run = object()
                client.finish.side_effect = lambda **kwargs: setattr(client, "run", None)
                wandb_logger = object.__new__(WandBLogger)
                wandb_logger.wandb = client
                other_logger = Mock()
                container = LoggerContainer()
                container.add_logger(wandb_logger)
                container.add_logger(other_logger)
                metrics = object.__new__(DynamicSeqMetricsProcessor)
                metrics.logger = container
                if failure:
                    try:
                        raise RuntimeError("simulated training failure")
                    except RuntimeError:
                        metrics.close()
                    client.finish.assert_called_once_with(exit_code=1)
                else:
                    metrics.close()
                    client.finish.assert_called_once_with()
                other_logger.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
