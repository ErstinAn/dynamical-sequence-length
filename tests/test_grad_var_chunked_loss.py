"""Independent full-CE oracle for TorchTitan's actual chunked-loss path."""

import unittest

import torch
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss

from tests.test_grad_var_trainer import tiny_trainer


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(5, 5)
        self.lm_head = torch.nn.Linear(5, 11)

    def forward(self, inputs):
        # As in the production Trainer, the loss wrapper runs lm_head separately.
        return self.body(inputs).tanh()


class GradVarChunkedLossTest(unittest.TestCase):
    def test_chunked_ce_against_independent_full_ce_gradients(self):
        # Tiny matrix operations are substantially slower with server-wide CPU
        # thread pools. Restore both thread count and RNG after this test.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20)
                for k in (3, 5, 8):
                    with self.subTest(k=k):
                        model = TinyLanguageModel()
                        trainer = tiny_trainer(k=k, b=2, seq=8, model=model)
                        trainer.loss_fn = ChunkedLossWrapper(
                            ChunkedLossWrapper.Config(
                                num_chunks=4,
                                loss_fn=CrossEntropyLoss.Config(),
                            )
                        )
                        trainer.loss_fn.set_lm_head(model.lm_head)
                        data = [
                            ({"input": torch.randn(2, 8, 5)}, torch.randint(0, 11, (2, 8)))
                            for _ in range(k)
                        ]

                        # Independently compute one full, unchunked CE gradient
                        # per group at fixed theta, with its OWN token count.
                        # autograd.grad leaves the training .grad fields empty.
                        samples = []
                        for inputs, labels in data:
                            logits = model.lm_head(model(inputs["input"]))
                            loss = torch.nn.functional.cross_entropy(
                                logits.reshape(-1, 11), labels.flatten(), reduction="mean"
                            )
                            samples.append(torch.autograd.grad(loss, tuple(model.parameters())))
                        expected = {}
                        for index, (name, _) in enumerate(model.named_parameters()):
                            stacked = torch.stack([sample[index].double() for sample in samples])
                            expected[f"grad_var_monitor/grad_var_monitor_{name}_var"] = (
                                (stacked - stacked.mean(0)).square().sum() / (k - 1)
                            ).item()
                        expected["global_grad_var"] = sum(expected.values())

                        # Executes the production Trainer/backward and monitor,
                        # including all internal lm_head chunk backwards.
                        trainer.train_step(iter(data))
                        actual, step = trainer.metrics_processor.logger.log.call_args.args
                        self.assertEqual(step, 1)
                        self.assertEqual(trainer.metrics_processor.logger.log.call_count, 1)
                        actual_variance = {
                            name: value
                            for name, value in actual.items()
                            if name == "global_grad_var" or name.startswith("grad_var_monitor/")
                        }
                        self.assertEqual(set(actual_variance), set(expected))
                        for name, value in expected.items():
                            torch.testing.assert_close(
                                torch.tensor(actual[name], dtype=torch.float64),
                                torch.tensor(value, dtype=torch.float64),
                                rtol=1e-6,
                                atol=1e-10,
                                msg=lambda message, name=name: f"K={k}, {name}: {message}",
                            )
                        self.assertAlmostEqual(
                            actual["global_grad_var"],
                            sum(
                                value
                                for name, value in actual_variance.items()
                                if name != "global_grad_var"
                            ),
                            places=14,
                        )
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
