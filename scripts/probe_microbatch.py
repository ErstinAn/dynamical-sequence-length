#!/usr/bin/env python3
"""Measure native Llama3 1B microbatches with the real training data path.

Each candidate runs in a separate single-GPU process. The probe builds the
production model, loss, attention, mixed precision, and activation checkpoints.
It allocates AdamW state, then runs forward and backward without an update.
The distributed smoke must confirm the selected microbatches.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LENGTHS = (2048, 4096, 8192, 16384)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def allocate_adamw_state(optimizers, torch):
    """Allocate AdamW's persistent state without changing model parameters."""
    state_bytes = 0
    largest_group_bytes = 0
    for optimizer in optimizers:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("The probe currently supports AdamW only.")
        for group in optimizer.param_groups:
            if group.get("amsgrad") or group.get("capturable") or group.get("fused"):
                raise ValueError("The probe requires the production foreach AdamW settings.")
            group_bytes = 0
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                if state:
                    raise ValueError("The probe requires a new optimizer state.")
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
                size = parameter.numel() * parameter.element_size()
                state_bytes += 2 * size
                group_bytes += size
            largest_group_bytes = max(largest_group_bytes, group_bytes)
    # No optimizer update runs here. Reserve two full group tensors for the
    # foreach update's temporary buffers. The final smoke measures the update.
    return state_bytes, 2 * largest_group_bytes


def worker(args):
    import torch
    import torch.distributed as dist
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.loss import IGNORE_INDEX
    from torchtitan.config import ConfigManager
    from torchtitan.trainer import Trainer

    from torchtitan_dynamic_seq_len.run_config import config_digest, resolve_point

    result = dict(sequence_length=args.length, microbatch_size=args.microbatch,
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  target_dp_degree=args.dp_degree, measured_dp_degree=1,
                  optimizer_updates=0, training_checkpoints=0)
    trainer = None
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Each probe worker must use exactly one GPU.")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        free_before, capacity = torch.cuda.mem_get_info()
        result.update(gpu_name=properties.name, total_memory_bytes=capacity,
                      free_before_model_bytes=free_before, torch_version=torch.__version__,
                      cuda_version=torch.version.cuda)
        rendered, _, _, arguments = resolve_point(args.config, 0)
        config = ConfigManager().parse_args([
            "--module", "torchtitan_dynamic_seq_len.experiment", "--config", "llama3_1b",
            *arguments,
        ])
        config.validate_dynamic()
        if config.optimizer.method != "adamw":
            raise ValueError("The probe requires the smoke AdamW optimizer.")
        result["config_sha256"] = config_digest(rendered)
        result["dataset_path"] = config.dataloader.dataset_path
        result["model_flavor"] = config.model_spec.flavor
        result["model_definition"] = config.model_spec.model.to_dict()
        # Disable output side effects for this diagnostic. The training smoke
        # keeps its W&B, validator, and full checkpoint settings.
        config.dump_folder = str(args.worker_output.parent / args.worker_output.stem)
        config.metrics.enable_wandb = False
        config.metrics.enable_tensorboard = False
        config.validator.enable = False
        config.checkpoint = CheckpointManager.Config(enable=False)
        # Use the upstream Trainer constructor to retain the production model
        # and forward/backward path without the experiment's save policy.
        trainer = Trainer(config)
        trainer.dataloader.set_batch_shape(seq_len=args.length, microbatch_size=args.microbatch)
        state_bytes, scratch_bytes = allocate_adamw_state(trainer.optimizers.optimizers, torch)
        result.update(optimizer_state_bytes=state_bytes,
                      estimated_optimizer_temporary_bytes=scratch_bytes,
                      model_parameter_count=sum(p.numel() for p in trainer.model_parts[0].parameters()))
        iterator = iter(trainer.dataloader)
        batches = [next(iterator) for _ in range(args.repeats)]
        if any(tuple(labels.shape) != (args.microbatch, args.length) for _, labels in batches):
            raise ValueError("The dataset did not supply a complete probe microbatch.")
        valid = sum(int((labels != IGNORE_INDEX).sum()) for _, labels in batches)
        if valid <= 0:
            raise ValueError("The probe batches have no prediction targets.")
        denominator = torch.tensor(valid, dtype=torch.int64, device=trainer.device)
        trainer.optimizers.zero_grad()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses = []
        for input_dict, labels in batches:
            inputs = {key: value.to(trainer.device) if isinstance(value, torch.Tensor) else value
                      for key, value in input_dict.items()}
            loss = trainer.forward_backward_step(
                input_dict=inputs, labels=labels.to(trainer.device), global_valid_tokens=denominator)
            losses.append(float(loss.detach()))
            del loss, inputs
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()
        budget = min(capacity * (1 - args.headroom), free_before - capacity * args.headroom)
        safe = peak + scratch_bytes <= budget
        result.update(status="ok", recommended=safe, headroom_fraction=args.headroom,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=peak, reserved_after_bytes=torch.cuda.memory_reserved(),
                      free_after_bytes=torch.cuda.mem_get_info()[0],
                      seconds=time.perf_counter() - started,
                      nominal_probe_tokens=args.repeats * args.microbatch * args.length,
                      valid_probe_targets=valid, mean_loss=sum(losses),
                      probe_accumulation_groups=args.repeats,
                      projected_reserved_with_optimizer_temporaries_bytes=peak + scratch_bytes,
                      safe_budget_bytes=int(budget))
    except torch.OutOfMemoryError as error:
        result.update(status="oom", recommended=False, error=str(error))
    except Exception as error:
        result.update(status="error", recommended=False,
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.worker_output, result)
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def measure(args):
    from torchtitan_dynamic_seq_len.run_config import load_config

    global_batch = load_config(args.config)["fixed"]["training.global_batch_size"]
    if global_batch % args.dp_degree:
        raise ValueError(f"The data-parallel degree must divide the global batch size of {global_batch}.")
    rank_batch = global_batch // args.dp_degree
    candidates = []
    microbatch = 1
    while microbatch <= min(rank_batch, args.max_microbatch):
        if rank_batch % microbatch == 0:
            candidates.append(microbatch)
        microbatch *= 2
    output = args.output.resolve()
    details = output.parent / (output.stem + "-candidates")
    details.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  config=str(args.config.resolve()), target_dp_degree=args.dp_degree,
                  global_batch_size=global_batch, headroom_fraction=args.headroom,
                  measurements=[], microbatches={},
                  limitation="Single-GPU memory estimate. Confirm values with the distributed training smoke.")
    env = dict(os.environ)
    # Preserve the first GPU identifier from the allocation. Never expose GPUs
    # outside the allocation or assume physical GPU zero belongs to this job.
    if env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"].split(",")[0]
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    write_json(output, report)
    for length in LENGTHS:
        chosen = None
        for microbatch in candidates:
            record = details / f"length-{length}-microbatch-{microbatch}.json"
            logfile = record.with_suffix(".log")
            command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
                       "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0", str(Path(__file__).resolve()),
                       "--worker", "--config", str(args.config.resolve()), "--length", str(length),
                       "--microbatch", str(microbatch), "--worker-output", str(record),
                       "--dp-degree", str(args.dp_degree), "--headroom", str(args.headroom),
                       "--repeats", str(args.repeats)]
            with logfile.open("w") as stream:
                completed = subprocess.run(command, env=env, cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
            if not record.exists():
                raise RuntimeError(f"The probe did not produce {record}. Read {logfile}.")
            measurement = json.loads(record.read_text())
            measurement["log_path"] = str(logfile)
            report["measurements"].append(measurement)
            write_json(output, report)
            if completed.returncode or measurement["status"] == "error":
                raise RuntimeError(f"The probe failed. Read {logfile}.")
            print(json.dumps({key: measurement.get(key) for key in
                              ("sequence_length", "microbatch_size", "status", "recommended",
                               "peak_reserved_bytes", "total_memory_bytes")}), flush=True)
            if measurement["status"] != "ok" or not measurement["recommended"]:
                break
            chosen = microbatch
        if chosen is None:
            report["status"] = "failed"
            write_json(output, report)
            raise RuntimeError(f"No microbatch passed the memory reserve at length {length}.")
        report["microbatches"][str(length)] = chosen
        write_json(output, report)
    report["status"] = "passed"
    write_json(output, report)
    print(json.dumps(report["microbatches"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke.yaml")
    parser.add_argument("--output", type=Path, default=Path("microbatch_probe.json"))
    parser.add_argument("--dp-degree", type=int, default=4)
    parser.add_argument("--headroom", type=float, default=0.20)
    parser.add_argument("--max-microbatch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, choices=LENGTHS, help=argparse.SUPPRESS)
    parser.add_argument("--microbatch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dp_degree < 1 or args.max_microbatch < 1 or args.repeats < 2 or args.timeout_seconds < 1:
        parser.error("Use positive sizes and at least two accumulation groups.")
    if not 0.10 <= args.headroom < 0.75:
        parser.error("Set --headroom between 0.10 and 0.75, excluding 0.75.")
    if args.worker:
        if not args.length or not args.microbatch or not args.worker_output:
            parser.error("The worker needs a length, microbatch, and output path.")
        worker(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
