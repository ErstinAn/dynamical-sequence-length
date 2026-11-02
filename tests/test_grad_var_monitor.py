"""Independent stacked-gradient references for the online statistic."""

import unittest

import torch

from torchtitan_dynamic_seq_len.grad_var.monitor import GradVarMonitor


class GradVarMonitorTest(unittest.TestCase):
    def measure(self, samples):
        parameters = [torch.nn.Parameter(torch.zeros_like(h[0])) for h in samples]
        monitor = GradVarMonitor([(f"p{i}", p) for i, p in enumerate(parameters)], len(samples[0]))
        monitor.CHUNK_ELEMENTS = 3  # Exercise multiple chunks and final short chunk.
        monitor.begin()
        cumulative = [torch.zeros_like(p) for p in parameters]
        for j in range(monitor.k):
            for p, h, c in zip(parameters, samples, cumulative, strict=True):
                c.add_(h[j] / monitor.k)
                p.grad = c.clone()
            before = [p.grad.clone() for p in parameters]
            monitor.observe()
            for p, old in zip(parameters, before, strict=True):
                torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
        metrics = monitor.finish()
        for i, h in enumerate(samples):
            reference = ((h - h.mean(dim=0)).square().sum() / (len(h) - 1)).item()
            self.assertAlmostEqual(
                metrics[f"grad_var_monitor/grad_var_monitor_p{i}_var"],
                reference,
                delta=max(1e-10, abs(reference) * 1e-8),
            )
        self.assertEqual(
            metrics["global_grad_var"], sum(v for n, v in metrics.items() if n != "global_grad_var")
        )
        return metrics

    def test_stacked_reference_multiple_k_and_parameter_shapes(self):
        generator = torch.Generator().manual_seed(812)
        for k in (2, 4, 8):
            with self.subTest(k=k):
                self.measure(
                    [
                        torch.randn(k, 3, 5, dtype=torch.float64, generator=generator),
                        torch.randn(k, 7, dtype=torch.float64, generator=generator),
                    ]
                )

    def test_equal_norm_opposite_directions(self):
        metrics = self.measure([torch.tensor([[3.0, 4.0], [-3.0, -4.0]], dtype=torch.float64)])
        self.assertEqual(metrics["global_grad_var"], 50.0)

    def test_identical_samples_and_large_mean(self):
        self.assertEqual(
            self.measure([torch.ones(4, 5, dtype=torch.float64)])["global_grad_var"], 0
        )
        self.measure(
            [
                1e8
                + torch.tensor(
                    [[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]], dtype=torch.float64
                )
            ]
        )

    def test_shared_frozen_and_missing_gradients(self):
        p = torch.nn.Parameter(torch.ones(2))
        frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        monitor = GradVarMonitor([("_orig_mod.weight", p), ("alias", p), ("frozen", frozen)], 2)
        monitor.begin()
        monitor.observe()  # h1 = 0, absent gradient.
        p.grad = torch.tensor([1.0, -2.0])  # C2; h2 = [2, -4].
        monitor.observe()
        metrics = monitor.finish()
        self.assertEqual(
            metrics, {"global_grad_var": 10.0, "grad_var_monitor/grad_var_monitor_weight_var": 10.0}
        )
        snapshot = monitor.previous["weight"]
        monitor.begin()
        p.grad = None
        monitor.observe()
        monitor.observe()
        self.assertEqual(monitor.finish()["global_grad_var"], 0)
        self.assertIs(monitor.previous["weight"], snapshot)

    def test_invalid_samples_lifecycle_and_names(self):
        p = torch.nn.Parameter(torch.ones(2))
        with self.assertRaisesRegex(ValueError, ">= 2"):
            GradVarMonitor([("p", p)], 1)
        with self.assertRaisesRegex(ValueError, "collision"):
            GradVarMonitor([("p", p), ("_orig_mod.p", torch.nn.Parameter(torch.ones(2)))], 2)
        monitor = GradVarMonitor([("p", p)], 2)
        with self.assertRaises(RuntimeError):
            monitor.observe()
        monitor.begin()
        with self.assertRaises(RuntimeError):
            monitor.finish()
        for value in (float("nan"), float("inf")):
            monitor.begin()
            p.grad = torch.full_like(p, value)
            monitor.observe()
            p.grad = torch.zeros_like(p)
            monitor.observe()
            with self.assertRaisesRegex(ValueError, "nonfinite.*p"):
                monitor.finish()
            monitor.reset()
            self.assertEqual(monitor.count, 0)
            self.assertFalse(monitor.active)


if __name__ == "__main__":
    unittest.main()
