"""Two-GPU correctness smoke, launched with torchrun inside a GPU allocation.

Uses synthetic fixed batches and the production Trainer train_step, FSDP2,
TorchTitan SUM normalization, FP32/BF16, clipping, and AdamW. No corpus or W&B.
This is a verification harness, not an experiment configuration.
"""

import copy
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.distributed.parallel_dims import ParallelDims

from tests.test_grad_var_chunked_loss import TinyLanguageModel
from tests.test_grad_var_trainer import cpu_metrics, tiny_trainer
from torchtitan_dynamic_seq_len.grad_var_monitor import full_local_tensor


def run_case(param_dtype, *, chunked_ce=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    b, seq, k = (2, 8, 4) if chunked_ce else (4, 256, 4)
    torch.manual_seed(19)
    base = (
        TinyLanguageModel()
        if chunked_ce
        else torch.nn.Sequential(
            torch.nn.Linear(seq, seq), torch.nn.GELU(), torch.nn.Linear(seq, seq)
        )
    ).to(device)
    rng = torch.Generator().manual_seed(800 + rank)
    if chunked_ce:
        data = [
            (
                {"input": torch.randn(b, seq, 5, generator=rng)},
                torch.randint(0, 11, (b, seq), generator=rng),
            )
            for _ in range(k)
        ]
    else:
        data = [
            ({"input": torch.randn(b, seq, generator=rng)}, torch.randn(b, seq, generator=rng))
            for _ in range(k)
        ]
    mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("dp_replicate", "dp_shard"))

    def wrap_model(model):
        policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
        for child in model.children():
            if any(True for _ in child.parameters()):
                fully_shard(child, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        disable_fsdp_gradient_division(model)
        return model

    def chunked_loss(model):
        loss = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(
                num_chunks=4,
                loss_fn=CrossEntropyLoss.Config(),
            )
        )
        loss.set_lm_head(model.lm_head)
        return loss

    # Store separately normalized, DP-summed group gradients as an independent
    # oracle. K is a power of two, so BF16's change in loss scale is exact.
    reference_model = (
        wrap_model(copy.deepcopy(base)) if chunked_ce else copy.deepcopy(base).to(param_dtype)
    )
    reference_loss = chunked_loss(reference_model) if chunked_ce else None
    group_samples = []
    for inputs, labels in data:
        if chunked_ce:
            # Separate synchronized backward per group, zeroing between groups:
            # unlike the monitor, this oracle never differences accumulated grads.
            reference_model.zero_grad(set_to_none=True)
            pred = reference_model(inputs["input"].to(device))
            loss, _ = reference_loss(
                pred, labels.to(device), torch.tensor(b * seq * world, device=device)
            )
            loss.backward()
            full = [full_local_tensor(p.grad).clone() for p in reference_model.parameters()]
        else:
            pred = reference_model(inputs["input"].to(device=device, dtype=param_dtype)).float()
            loss = (pred - labels.to(device)).square().sum() / (b * seq * world)
            sample = torch.autograd.grad(loss, tuple(reference_model.parameters()))
            full = [g.float() for g in sample]
            for g in full:
                dist.all_reduce(g)
        group_samples.append(full)
    expected = {}
    for i, (name, _) in enumerate(reference_model.named_parameters()):
        h = torch.stack([sample[i].double() for sample in group_samples])
        expected[f"grad_var_monitor/grad_var_monitor_{name}_var"] = (
            (h - h.mean(0)).square().sum() / (k - 1)
        ).item()
    expected["global_grad_var"] = sum(expected.values())
    trainers = []
    measurements = []
    for enabled in (False, True):
        model = wrap_model(copy.deepcopy(base))
        trainer = tiny_trainer(False, k=k, b=b, seq=seq, model=model)
        trainer.device = device
        trainer.parallel_dims = ParallelDims(
            dp_replicate=world, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=world
        )
        trainer.metrics_processor = cpu_metrics(trainer.parallel_dims)
        trainer.config.training.global_batch_size = b * world * k
        trainer.config.grad_var_monitor.enable = enabled
        trainer._init_grad_var_monitor()
        trainer.loss_fn = (
            chunked_loss(model)
            if chunked_ce
            else lambda pred, labels, total: ((pred.float() - labels).square().sum() / total, None)
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trainer.train_step(iter(data))
        torch.cuda.synchronize()
        measurements.append(
            {
                "enabled": enabled,
                "first_step_seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        trainers.append(trainer)
    off, on = trainers
    payload = on.metrics_processor.logger.log.call_args.args[0]
    for key, value in expected.items():
        torch.testing.assert_close(
            torch.tensor(payload[key]),
            torch.tensor(value),
            rtol=2e-5,
            atol=1e-8,
            msg=lambda msg: f"{param_dtype} {key}: {msg}",
        )
    for p, q in zip(off.model_parts[0].parameters(), on.model_parts[0].parameters(), strict=True):
        torch.testing.assert_close(full_local_tensor(p), full_local_tensor(q), rtol=0, atol=0)
        torch.testing.assert_close(
            full_local_tensor(p.grad), full_local_tensor(q.grad), rtol=0, atol=0
        )
    # Warmed steps at the same successive theta values; compare after both paths.
    for trainer, measurement in zip(trainers, measurements, strict=True):
        elapsed = []
        # Measure incremental peak from each path's live baseline. Snapshot memory
        # is reported separately, independent of retained comparison models.
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        for step in range(2, 11):
            trainer.step = step
            trainer.metrics_processor.data_loading_times = [0.001]
            torch.cuda.synchronize()
            start = time.perf_counter()
            trainer.train_step(iter(data))
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
        measurement["warmed_step_seconds"] = sum(elapsed) / len(elapsed)
        measurement["incremental_peak_bytes"] = torch.cuda.max_memory_allocated() - baseline_bytes
        measurement["snapshot_bytes"] = (
            sum(t.numel() * t.element_size() for t in trainer.grad_var_monitor.previous.values())
            if trainer.grad_var_monitor
            else 0
        )
    for p, q in zip(off.model_parts[0].parameters(), on.model_parts[0].parameters(), strict=True):
        torch.testing.assert_close(full_local_tensor(p), full_local_tensor(q), rtol=0, atol=0)
    gathered = [None] * world
    dist.all_gather_object(gathered, payload["global_grad_var"])
    assert gathered == [gathered[0]] * world, gathered
    if rank == 0:
        print(
            json.dumps(
                {
                    "dtype": str(param_dtype),
                    "chunked_ce": chunked_ce,
                    "world_size": world,
                    "expected_var": expected["global_grad_var"],
                    "actual_var": payload["global_grad_var"],
                    "measurements": measurements,
                }
            ),
            flush=True,
        )


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("launch this smoke with two GPUs")
        torch.backends.cuda.matmul.allow_tf32 = False
        run_case(torch.float32)
        run_case(torch.bfloat16)
        run_case(torch.float32, chunked_ce=True)
        run_case(torch.bfloat16, chunked_ce=True)
        if dist.get_rank() == 0:
            print(
                "PASS: pre-clipping variance oracle, per-parameter sum, DP replicas, and exact updates",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
