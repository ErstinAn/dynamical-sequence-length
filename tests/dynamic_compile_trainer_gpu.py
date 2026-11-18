"""Check the 100M Trainer at compile boundaries on allocated GPUs."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger

from torchtitan_dynamic_seq_len.run_config import resolve_point

ROOT = Path(__file__).resolve().parents[1]


class CaptureLogger:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.last = None

    def log(self, values, step):
        self.last = values
        self.wrapped.log(values, step)

    def close(self):
        self.wrapped.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw, _, _, options = resolve_point(
        ROOT / "legacy/configs/dynamic_seq_bs/linear_bs_cyclic_seq_constant_lr.yaml", 0
    )
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
    cfg.dataloader.dataset = "sequence_length_smoke"
    cfg.dataloader.dataset_path = str(ROOT / "tests/fixtures/text.jsonl")
    cfg.metrics.enable_wandb = False
    cfg.metrics.enable_tensorboard = False
    cfg.checkpoint.enable = False
    cfg.dump_folder = str(args.output / "run")
    init_logger()
    trainer = cfg.build()
    capture = CaptureLogger(trainer.metrics_processor.logger)
    trainer.metrics_processor.logger = capture
    steps = [
        1,
        2,
        8,
        20,
        31,
        61,
        100,
        128,
        129,
        200,
        240,
        256,
        257,
        480,
        512,
        768,
        1024,
        1400,
        1500,
        1520,
        1528,
        1529,
        1530,
        1536,
        1664,
        1920,
        2560,
        3200,
        3712,
        3840,
    ]
    rows = []
    total = 0
    rank = int(os.environ.get("RANK", "0"))

    def batches():
        while True:
            b, s = trainer.current_microbatch_size, trainer.current_seq_len
            stream = torch.randint(3, 32000, (b, s + 1))
            positions = torch.arange(s).expand(b, -1).contiguous()
            yield (
                {"input": stream[:, :-1].contiguous(), "positions": positions},
                stream[:, 1:].contiguous(),
            )

    iterator = trainer.batch_generator(batches())
    try:
        for step in steps:
            trainer.step = step
            trainer.train_step(iterator)
            torch.cuda.synchronize()
            metrics = capture.last
            b, s = trainer.current_global_batch_size, trainer.current_seq_len
            total += b * s
            assert metrics["n_tokens_seen"] == total
            assert metrics["n_valid_tokens_seen"] == total
            assert math.isfinite(metrics["loss_metrics/global_avg_loss"])
            assert math.isfinite(metrics["grad_norm"])
            rows.append(
                {
                    "step": step,
                    "seq_len": s,
                    "global_batch": b,
                    "microbatch": trainer.current_microbatch_size,
                    "loss": metrics["loss_metrics/global_avg_loss"],
                    "tokens": total,
                    "active_gib": metrics["memory/max_active(GiB)"],
                }
            )
            if step in (20, 1520, 3840):
                trainer.validator.validate(trainer.model_parts, step)
                assert capture.last["validation_metrics/evaluated_tokens"] == 20971520
            if rank == 0:
                (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
                print("COMPILE_TRAINER_STEP " + json.dumps(rows[-1]), flush=True)
        if rank == 0:
            result = dict(
                status="passed",
                gpu=torch.cuda.get_device_name(),
                world_size=int(os.environ["WORLD_SIZE"]),
                steps=steps,
                rows=rows,
                graphs=dict(torch._dynamo.utils.counters["stats"]),
            )
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception as error:
        (args.output / f"failure-rank{rank}.json").write_text(
            json.dumps(
                {
                    "exception": type(error).__name__,
                    "message": str(error),
                    "completed_steps": len(rows),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
