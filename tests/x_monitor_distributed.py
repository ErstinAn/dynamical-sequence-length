"""Run with torchrun; compare SUM DP and accumulation against a full-batch reference."""

import os
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from tests.test_x_monitor import SyntheticSampler, x_trainer
from torchtitan_dynamic_seq_len.grad_var.monitor import XMonitor
from torchtitan_dynamic_seq_len.grad_var.probes import compute_group_gradient


def main():
    cuda = os.environ.get("X_TEST_CUDA") == "1"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(seconds=120))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(123)
        model = torch.nn.Linear(3, 3).double().to(device)
        trainer = x_trainer(model)
        trainer.model_parts = [model]
        trainer.device = device

        # Explicit SUM at the group boundary is the numerical reference for
        # the pinned FSDP2 reduction, which is exercised by the GPU smoke run.
        def synchronize():
            for p in model.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        trainer._check_gradient_sync = synchronize
        trainer._grad_var_group = dist.group.WORLD
        trainer._x_dp_rank, trainer._x_dp_degree = rank, world
        cfg = trainer.config.grad_var_monitor
        cfg.group_sequences = 16
        trainer.grad_var_monitor = XMonitor(model.named_parameters(), 3)
        references = []
        for local_batch_size in (1, 2, 4):
            cfg.local_batch_size = local_batch_size
            trainer.grad_var_monitor.begin()
            references = []
            for group in range(3):
                anchors = np.arange(16) + group * 16
                compute_group_gradient(trainer, anchors, 3)
                actual = [p.grad.clone() for p in model.parameters()]
                inputs, labels = next(SyntheticSampler().microbatches(anchors, 3, 0, 1, 16))
                reference = torch.autograd.grad(
                    (model(inputs["input"].to(device)) - labels.to(device)).square().mean(),
                    tuple(model.parameters()),
                )
                references.append(torch.cat([g.flatten() for g in reference]))
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
            samples = torch.stack(references)
            expected = 48 * (samples - samples.mean(0)).square().sum() / 2
            actual = trainer.grad_var_monitor.finish(48)
            torch.testing.assert_close(
                torch.tensor(actual, dtype=torch.float64, device=device),
                expected,
                atol=1e-10,
                rtol=1e-12,
            )
        if rank == 0:
            print(
                f"X distributed gradient normalization PASS: DP={world}, device={device.type}, microbatches=1,2,4",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
