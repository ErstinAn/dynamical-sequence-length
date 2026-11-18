"""Run a test YAML or selected schedule positions with real model/data and W&B."""

import argparse
import json
import math
import os
import uuid
from pathlib import Path

import torch.distributed as dist
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger

from torchtitan_dynamic_seq_len.run_config import resolve_point


class CaptureLogger:
    def __init__(self, wrapped, output):
        self.wrapped, self.output, self.rows = wrapped, output, []
        self._loggers = getattr(wrapped, "_loggers", [])

    def log(self, values, step):
        self.wrapped.log(values, step)
        if dist.get_rank() == 0:
            self.rows.append({"step": step, **values})
            self.output.write_text(json.dumps(self.rows, indent=2))

    def close(self):
        self.wrapped.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule-steps", type=int, nargs="+")
    parser.add_argument("--resume-step", type=int)
    args = parser.parse_args()
    raw, _, _, options = resolve_point(args.config, 0)
    if args.schedule_steps and any(
        s < 1 or s > raw["fixed"]["training.steps"] for s in args.schedule_steps
    ):
        raise ValueError("Selected steps must be within configured schedule")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["DYNAMIC_SEQ_LEN_VALIDATION_DIR"] = raw["paths"]["validation_dir"]
    cfg = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_dynamic_seq_len.config_registry",
            "--config",
            raw["torchtitan_config"],
            *options,
        ]
    )
    # Every invocation is a debug test, even when sampling the prepared formal YAML.
    run_id = os.environ.get("LOSS_GRAD_TEST_RUN_ID") or uuid.uuid4().hex
    os.environ.update(
        WANDB_PROJECT="dynamic-seq-bs-debug",
        WANDB_ENTITY="dynamic_seq_length",
        WANDB_TEAM="dynamic_seq_length",
        WANDB_RUN_ID=run_id,
        WANDB_RESUME="never",
        WANDB_MODE="online",
        WANDB_RUN_NAME=raw["name"] + ("-selected-steps-test" if args.schedule_steps else ""),
    )
    reference_rows = None
    if args.resume_step is not None:
        cfg.checkpoint.load_step = args.resume_step
        reference_rows = json.loads((args.output / "metrics.json").read_text())
    cfg.metrics.enable_wandb = True
    cfg.dump_folder = str(args.output / "run")
    if args.schedule_steps:
        cfg.checkpoint.enable = False
        cfg.validator.enable = False
    init_logger()
    trainer = cfg.build()
    metric_file = "metrics-resumed.json" if args.resume_step is not None else "metrics.json"
    capture = CaptureLogger(trainer.metrics_processor.logger, args.output / metric_file)
    trainer.metrics_processor.logger = capture
    try:
        if args.schedule_steps:
            iterator = trainer.batch_generator(trainer.dataloader)
            for step in args.schedule_steps:
                trainer.step = step
                trainer.train_step(iterator)
        else:
            trainer.train()
        if dist.get_rank() == 0:
            rows = [row for row in capture.rows if "loss_metrics/long_avg_loss" in row]
            expected_steps = args.schedule_steps or list(range((args.resume_step or 0) + 1, cfg.training.steps + 1))
            assert [row["step"] for row in rows] == expected_steps
            for row in rows:
                assert all(
                    math.isfinite(row[key])
                    for key in (
                        "loss_metrics/long_avg_loss",
                        "loss_metrics/short_seq_loss",
                        "grad_norm",
                    )
                ), row
                assert row["loss_grad/grad_seq_length"] <= row["loss_grad/loss_seq_length"]
                plan = cfg.plan(trainer.schedules, row["step"], trainer.parallel_dims.dp_replicate)
                assert row["loss_grad/grad_microbatch_size"] == plan.grad_microbatch
                assert row["loss_grad/loss_microbatch_size"] == plan.loss_microbatch
            validation_rows = [row for row in capture.rows if "validation_metrics/loss" in row]
            if cfg.validator.enable:
                assert validation_rows, "Enabled validation did not produce any metrics"
                for row in validation_rows:
                    length = trainer.schedules["loss_seq_len_scheduler"].length_for_step(
                        row["step"]
                    )
                    assert row["validation_metrics/seq_len"] == length
                    assert math.isfinite(row["validation_metrics/loss"])
            if reference_rows is not None:
                previous = {r["step"]: r for r in reference_rows if "loss_metrics/long_avg_loss" in r}
                for row in rows:
                    for key in ("loss_metrics/long_avg_loss", "loss_metrics/short_seq_loss", "grad_norm"):
                        assert math.isclose(row[key], previous[row["step"]][key], rel_tol=2e-5, abs_tol=2e-5), (key, row)
            result = {
                "status": "passed",
                "steps": expected_steps,
                "wandb_url": f"https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/{run_id}",
                "config": str(args.config),
                "selected_steps_only": bool(args.schedule_steps),
                "global_batch_sizes": sorted({row["loss_grad/global_batch_size"] for row in rows}),
                "grad_lengths": [row["loss_grad/grad_seq_length"] for row in rows],
                "grad_microbatches": [row["loss_grad/grad_microbatch_size"] for row in rows],
                "loss_microbatches": [row["loss_grad/loss_microbatch_size"] for row in rows],
                "max_memory_gib": max(row["memory/max_active(GiB)"] for row in rows),
                "validation": getattr(getattr(trainer, "validator", None), "last_result", None),
                "validation_steps": [row["step"] for row in validation_rows],
                "validation_lengths": [
                    row["validation_metrics/seq_len"] for row in validation_rows
                ],
            }
            result["resume_step"] = args.resume_step
            result["document_masking"] = cfg.document_masking
            result_name = "result-resumed.json" if args.resume_step is not None else "result.json"
            (args.output / result_name).write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
    finally:
        trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
