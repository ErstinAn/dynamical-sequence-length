"""Opt-in four-rank Gloo tests; no GPUs, data downloads, or W&B writes."""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from tests.test_grad_var_trainer import tiny_trainer
from torchtitan_dynamic_seq_len.grad_var.monitor import GradVarMonitor, full_local_tensor


def _worker(rank, world, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    try:
        # The user's 512 / (32 * 4) = 4 example, with fixed sequence length.
        k, b, seq = 4, 32, 3
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        monitor = GradVarMonitor([("p", p)], k)
        monitor.begin()
        cumulative = torch.zeros_like(p)
        groups = []
        for j in range(k):
            local = torch.tensor([(-1.0) ** rank * (j + 1), rank + j * j], dtype=torch.float64)
            # local is a local *mean*; emulate global-token-normalized backward.
            contribution = local / (world * k)
            dist.all_reduce(contribution)
            cumulative.add_(contribution)
            p.grad = cumulative.clone()
            monitor.observe()
            # Independently gather full vectors; never average rank norms.
            gathered = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(gathered, local)
            groups.append(torch.stack(gathered).mean(0))
        stacked = torch.stack(groups)
        expected = ((stacked - stacked.mean(0)).square().sum() / (k - 1)).item()
        actual = monitor.finish()["global_grad_var"]
        assert abs(actual - expected) < 1e-10, (actual, expected)
        results = [None] * world
        dist.all_gather_object(results, actual)
        assert results == [results[0]] * world

        mesh = init_device_mesh("cpu", (world, 1), mesh_dim_names=("replica", "shard"))
        full = distribute_tensor(torch.arange(4.0), mesh, [Replicate(), Shard(0)])
        torch.testing.assert_close(full_local_tensor(full), torch.arange(4.0))
        sharded = distribute_tensor(torch.arange(4.0), mesh, [Shard(0), Replicate()])
        try:
            full_local_tensor(sharded)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a genuinely sharded gradient")

        # Error flags and unequal token counts must fail on every rank, with no
        # pending metric or optimizer update. Rank 0 alone injects the bad state.
        for mode in ("valid", "unequal_tokens", "rank_error"):
            trainer = tiny_trainer(k=k, b=b, seq=seq)
            trainer.config.training.global_batch_size = 512
            trainer._grad_var_fixed_batch = 512
            trainer._grad_var_group = dist.group.WORLD
            trainer._grad_var_error = (
                "injected bad gradient" if mode == "rank_error" and rank == 0 else None
            )
            trainer._grad_var_counts = [torch.tensor(b * seq, dtype=torch.int64) for _ in range(k)]
            if mode == "unequal_tokens" and rank == 0:
                trainer._grad_var_counts[0] -= 1
            trainer.grad_var_monitor.begin()
            for _ in range(k):
                trainer.grad_var_monitor.observe()  # missing grads -> zero, valid input
            total = torch.stack(trainer._grad_var_counts).sum()
            dist.all_reduce(total)
            failed = False
            try:
                trainer._finish_grad_var(total)
            except RuntimeError:
                failed = True
            assert failed == (mode != "valid"), (rank, mode)
            assert bool(trainer.metrics_processor._grad_var_pending) == (mode == "valid")
            assert len(trainer.optimizers.gradients) == 0
    finally:
        dist.destroy_process_group()


class GradVarDistributedTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("GRAD_VAR_DISTRIBUTED_TEST") == "1", "opt in to local Gloo sockets"
    )
    def test_four_rank_vectors_tokens_errors_and_dtensor(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(_worker, args=(4, str(Path(root) / "rendezvous")), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
