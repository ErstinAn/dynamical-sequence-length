"""Measure ordinary and monitored steps with real parent batches on allocated GPUs."""

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger

from torchtitan_dynamic_seq_len.run_config import resolve_point


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class CaptureLogger:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.last = None

    def log(self, metrics, step):
        self.last = dict(metrics)
        self.wrapped.log(metrics, step)

    def close(self):
        self.wrapped.close()


def summarize(rows):
    result = {}
    for phase in ("ordinary", "monitor"):
        measured = [row for row in rows if row["phase"] == phase and not row["warmup"]]
        result[phase] = {
            "samples": len(measured),
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in measured),
            "min_wall_seconds": min(row["wall_seconds"] for row in measured),
            "max_wall_seconds": max(row["wall_seconds"] for row in measured),
            "median_probe_seconds": statistics.median(row["probe_seconds"] for row in measured),
            "peak_active_gib": max(row["peak_active_gib"] for row in measured),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in measured),
        }
    base = result["ordinary"]["median_wall_seconds"]
    extra = max(0.0, result["monitor"]["median_wall_seconds"] - base)
    result["projected_hours_20tpp_with_45min_margin"] = {
        "interval20_398_monitors": (7637 * base + 398 * extra) / 3600 + 0.75,
        "interval50_172_monitors": (7637 * base + 172 * extra) / 3600 + 0.75,
        "interval100_97_monitors": (7637 * base + 97 * extra) / 3600 + 0.75,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    init_logger()
    raw, _, _, cli = resolve_point(args.config, 0)
    os.environ["DYNAMIC_SEQ_LEN_VALIDATION_DIR"] = raw["paths"]["validation_dir"]
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_dynamic_seq_len.config_registry",
            "--config",
            raw["torchtitan_config"],
            *cli,
        ]
    )
    config.training.local_batch_size = args.microbatch
    config.training.steps = 27
    config.checkpoint.enable = False
    config.debug.print_config = False
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = True
    config.metrics.profile_freq = 0
    config.dump_folder = str(args.output / "run")
    config.validate_seq_grad()
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2 and 128 % (world * args.microbatch) == 0
    assert config.training.global_batch_size == 128 and config.training.seq_len == 16384
    if args.preflight:
        print(
            json.dumps(
                {
                    "microbatch": args.microbatch,
                    "accumulation": 128 // (world * args.microbatch),
                    "config": "valid",
                }
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = None
    rank = int(os.environ.get("RANK", "0"))
    rows = []
    try:
        trainer = config.build()
        capture = CaptureLogger(trainer.metrics_processor.logger)
        trainer.metrics_processor.logger = capture
        iterator = trainer.batch_generator(trainer.dataloader)
        # Use the actual train_step without the outer loop's validation or saves.
        # Every step still reads new [128,16384] data, including ordinary steps.
        phases = [("ordinary", True)] * 3 + [("ordinary", False)] * 20
        phases += [("monitor", True)] + [("monitor", False)] * 3
        for step, (phase, warmup) in enumerate(phases, start=1):
            trainer.step = step
            config.seq_grad.enable = phase == "monitor"
            trainer.gc_handler.run(step)
            dist.barrier()
            torch.cuda.synchronize(trainer.device)
            started = time.perf_counter()
            trainer.train_step(iterator)
            torch.cuda.synchronize(trainer.device)
            wall = time.perf_counter() - started
            values = capture.last
            assert values is not None and values["n_tokens_seen"] == step * 128 * 2048
            assert math.isfinite(values["loss_metrics/global_avg_loss"])
            assert math.isfinite(values["grad_norm"])
            if phase == "monitor":
                for length in config.seq_grad.lengths:
                    for name in ("distance", "relative_distance", "grad_norm", "cosine"):
                        assert math.isfinite(values[f"seq_grad_{name}/s{length:05d}"])
                assert values["seq_grad_reference/valid_tokens"] == 128 * 16384
            measured = torch.tensor(
                [
                    wall,
                    values["time_metrics/end_to_end(s)"],
                    values["seq_grad_cost/probe_seconds"],
                    values["memory/max_active(GiB)"],
                    values["memory/max_reserved(GiB)"],
                    values["memory/num_ooms"],
                    values["memory/num_alloc_retries"],
                    values["time_metrics/data_loading(s)"] * trainer.gradient_accumulation_steps,
                ],
                dtype=torch.float64,
                device=trainer.device,
            )
            dist.all_reduce(measured, op=dist.ReduceOp.MAX)
            numbers = measured.tolist()
            assert numbers[5] == 0, "CUDA OOM recorded"
            row = dict(
                zip(
                    [
                        "wall_seconds",
                        "internal_step_seconds",
                        "probe_seconds",
                        "peak_active_gib",
                        "peak_reserved_gib",
                        "oom_count",
                        "allocation_retries",
                        "data_loading_seconds",
                    ],
                    numbers,
                    strict=True,
                )
            )
            row.update(
                step=step,
                phase=phase,
                warmup=warmup,
                loss=values["loss_metrics/global_avg_loss"],
                grad_norm=values["grad_norm"],
            )
            rows.append(row)
            if rank == 0:
                atomic_json(
                    args.output / "progress.json", {"microbatch": args.microbatch, "rows": rows}
                )
                print("MICROBATCH_STEP " + json.dumps(row), flush=True)
        if rank == 0:
            result = dict(
                status="passed",
                microbatch=args.microbatch,
                world_size=world,
                gradient_accumulation_steps=trainer.gradient_accumulation_steps,
                gpu=torch.cuda.get_device_name(trainer.device),
                gpu_total_gib=torch.cuda.get_device_properties(trainer.device).total_memory / 2**30,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                warmup_excluded=True,
                validation_and_checkpoint_excluded=True,
                wandb_disabled_for_benchmark=True,
                summary=summarize(rows),
                rows=rows,
            )
            atomic_json(args.output / "result.json", result)
            print(
                "MICROBATCH_RESULT " + json.dumps({k: v for k, v in result.items() if k != "rows"}),
                flush=True,
            )
    except Exception as error:
        atomic_json(
            args.output / f"failure_rank{rank}.json",
            {
                "microbatch": args.microbatch,
                "exception": type(error).__name__,
                "message": str(error)[:2000],
                "completed_steps": len(rows),
            },
        )
        raise
    else:
        trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
