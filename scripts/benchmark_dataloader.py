#!/usr/bin/env python3
"""Compare data loading with the TorchTitan Trainer on one compute node.

The parent starts a new torchrun process for each case.
It reverses the case order for odd repetitions.
The replay mode uses cached batches on the CPU.
It still transfers data from the CPU to the GPU.
"""

import argparse
import itertools
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torchtitan_dynamic_seq_len.run_config import load_config, resolve_point, source_digest


def run_case(args):
    import torch
    import torch.distributed as dist
    from torchtitan.config import ConfigManager
    from torchtitan.tools.logging import init_logger

    raw, _, _, cli = resolve_point(args.config, 0)
    os.environ["DYNAMIC_SEQ_LEN_VALIDATION_DIR"] = raw["paths"]["validation_dir"]
    for key, name in (("hf_home", "HF_HOME"), ("torch_home", "TORCH_HOME")):
        if raw["paths"].get(key):
            os.environ[name] = raw["paths"][key]
    init_logger()
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_dynamic_seq_len.config_registry",
            "--config",
            raw["torchtitan_config"],
            *cli,
        ]
    )
    mode, length, workers, repeat = args.case
    length, workers, repeat = int(length), int(workers), int(repeat)
    spec = raw["benchmark"]
    warmup, steps = spec["warmup_steps"], spec["measure_steps"]
    config.training.seq_len = length
    config.training.steps = warmup + steps
    config.dataloader.num_workers = workers
    config.dataloader.prefetch_factor = 2 if workers else None
    config.dataloader.persistent_workers = workers > 0
    config.validator.enable = config.checkpoint.enable = False
    config.metrics.enable_wandb = False
    config.metrics.log_freq = 1
    config.dump_folder = str(args.output / f"{mode}-L{length}-w{workers}-r{repeat}")
    trainer = config.build()
    try:
        source = trainer.dataloader
        replay_bytes = 0
        if mode == "replay":
            iterator = iter(source)
            cached = [next(iterator) for _ in range(2 * trainer.gradient_accumulation_steps)]
            replay_bytes = sum(
                tensor.numel() * tensor.element_size()
                for inputs, labels in cached
                for tensor in [*inputs.values(), labels]
            )
            # The TorchTitan Trainer changes the input dictionary during data transfer.
            source = ((dict(inputs), labels) for inputs, labels in itertools.cycle(cached))
        data = trainer.batch_generator(source)
        durations, loading, tokens = [], [], []
        original_log = trainer.metrics_processor.logger.log
        latest = {}

        def capture(payload, step):
            latest.clear()
            latest.update(payload)
            original_log(payload, step)

        trainer.metrics_processor.logger.log = capture
        for step in range(1, warmup + steps + 1):
            if step == warmup + 1:
                torch.cuda.reset_peak_memory_stats()
            trainer.step = step
            trainer.gc_handler.run(step)
            dist.barrier()
            torch.cuda.synchronize()
            before = trainer.ntokens_seen
            start = time.perf_counter()
            trainer.train_step(data)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step > warmup:
                # The slowest rank sets the step time for all ranks.
                timing = torch.tensor([elapsed], device=trainer.device, dtype=torch.float64)
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                durations.append(timing.item())
                loading.append(latest["time_metrics/data_loading(s)"])
                count = torch.tensor(trainer.ntokens_seen - before, device=trainer.device)
                dist.all_reduce(count)
                tokens.append(count.item())
        result = {
            "mode": mode,
            "seq_len": length,
            "num_workers": workers,
            "repeat": repeat,
            "gpu": torch.cuda.get_device_name(),
            "world_size": dist.get_world_size(),
            "local_batch_size": config.training.local_batch_size,
            "accumulation": trainer.gradient_accumulation_steps,
            "warmup_steps": warmup,
            "measure_steps": steps,
            "compile": config.compile.enable,
            "dtype": config.training.mixed_precision_param,
            "cpu_affinity_count": len(os.sched_getaffinity(0)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "step_median_s": statistics.median(durations),
            "step_p95_s": sorted(durations)[min(len(durations) - 1, int(0.95 * len(durations)))],
            "global_tokens_per_second": sum(tokens) / sum(durations),
            "mean_load_seconds_per_microbatch": statistics.mean(loading),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_main_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "replay_cpu_bytes": replay_bytes,
            "step_seconds": durations,
            "cache_note": "OS/HF caches are not flushed; compare reversed-order repeats.",
            "source_sha256": source_digest(),
        }
        if dist.get_rank() == 0:
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "legacy/configs/benchmark_dataloader.yaml")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dataloader-benchmark")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", nargs=4, metavar=("MODE", "LENGTH", "WORKERS", "REPEAT"))
    args = parser.parse_args()
    spec = load_config(args.config)["benchmark"]
    if min(args.nproc, spec["warmup_steps"], spec["measure_steps"], spec["repeats"]) < 1:
        parser.error("GPU count, warmup, measured steps and repeats must be positive")
    if args.case:
        args.output.mkdir(parents=True, exist_ok=True)
        return run_case(args)
    cases = [
        ("pipeline", length, workers) for length in spec["lengths"] for workers in spec["workers"]
    ]
    cases += [("replay", length, 0) for length in spec["lengths"]]
    for repeat in range(spec["repeats"]):
        for mode, length, workers in cases[:: (-1 if repeat % 2 else 1)]:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.nproc}",
                str(Path(__file__).resolve()),
                "--config",
                str(args.config.resolve()),
                "--output",
                str(args.output.resolve()),
                "--case",
                mode,
                str(length),
                str(workers),
                str(repeat),
            ]
            if args.dry_run:
                print(json.dumps(command))
            else:
                subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
