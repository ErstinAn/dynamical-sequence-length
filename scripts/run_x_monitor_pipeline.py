#!/usr/bin/env python3
"""On an allocated node, accept the X monitor before starting its full trajectory."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cluster/vt_arc"))


def run_stage(stage, logfile):
    env = os.environ.copy()
    # Public API reads can start a service in this parent process. Training runs
    # must own their service lifecycle, including after the smoke process exits.
    env.pop("WANDB_SERVICE", None)
    with logfile.open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "cluster/vt_arc/run_from_yaml.py"),
                stage["config"],
                "--manifest",
                stage["manifest"],
                "--index",
                "0",
            ],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    print(f"{stage['name']}: exit={result.returncode}, log={logfile}", flush=True)
    return result.returncode


def validate_local_artifacts(stage):
    config = yaml.safe_load(Path(stage["config"]).read_text())
    fixed = config["fixed"]
    events = [
        json.loads(line)
        for line in (Path(stage["dump_folder"]) / "x_events.jsonl").read_text().splitlines()
    ]
    lengths = fixed["grad_var_monitor.lengths"]
    keys = {"grad_var/step"} | {f"grad_var/x_s{s}" for s in lengths}
    steps = ([0] if fixed["grad_var_monitor.include_initial"] else []) + list(
        range(
            fixed["grad_var_monitor.interval"],
            fixed["training.steps"] + 1,
            fixed["grad_var_monitor.interval"],
        )
    )
    if [e["metrics"]["grad_var/step"] for e in events] != steps:
        raise RuntimeError(f"Smoke events must be {steps} exactly")
    for event in events:
        if set(event["metrics"]) != keys or not all(
            math.isfinite(v) for v in event["metrics"].values()
        ):
            raise RuntimeError("Incorrect or nonfinite smoke payload")
    block_size = fixed.get("grad_var_monitor.replicate_groups", 0)
    if block_size:
        repeats = [
            json.loads(line)
            for line in (Path(stage["dump_folder"]) / "x_replicates.jsonl").read_text().splitlines()
        ]
        if [row["step"] for row in repeats] != steps:
            raise RuntimeError("Missing independent replicate events")
        for row in repeats:
            if (
                row["group_sequences"] != fixed["grad_var_monitor.group_sequences"]
                or row["groups_per_replicate"] != block_size
                or len(row["replicates"]) != fixed["grad_var_monitor.num_groups"] // block_size
            ):
                raise RuntimeError("Incorrect independent replicate groups")
            for replicate in row["replicates"]:
                if set(replicate) != keys - {"grad_var/step"} or not all(
                    math.isfinite(v) and v >= 0 for v in replicate.values()
                ):
                    raise RuntimeError("Incorrect or nonfinite replicate payload")
    return events, config, keys, steps


def accept(stage):
    events, config, keys, steps = validate_local_artifacts(stage)
    # The process has finished and W&B has flushed its final open step.
    import wandb

    settings = config["wandb"]
    run = wandb.Api(timeout=60).run(
        f"{settings['entity']}/{settings['project']}/{stage['wandb_id']}"
    )
    history = list(run.scan_history(keys=sorted(keys)))
    if [row["grad_var/step"] for row in history] != steps or any(
        not all(math.isfinite(row[key]) for key in keys) for row in history
    ):
        raise RuntimeError("W&B did not receive all smoke events; full run withheld")
    validation = list(
        run.scan_history(keys=["validation_metrics/loss", "validation_metrics/evaluated_tokens"])
    )
    if not validation or any(
        not math.isfinite(row["validation_metrics/loss"])
        or row["validation_metrics/evaluated_tokens"] != config["fixed"]["validator.token_budget"]
        for row in validation
    ):
        raise RuntimeError("Smoke must finish the full-budget validation before training")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = args.plan.parent
    env = dict(os.environ, X_TEST_CUDA="1", OMP_NUM_THREADS="1")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "-m",
            "tests.x_monitor_distributed",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    if plan.get("check_packed_attention", False):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.packing_attention_gpu",
                "--output",
                str(output / "packed_attention_check.json"),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    sizes = plan.get("probe_microbatch_sizes", [32, 16])
    for index, size in enumerate(sizes):
        stages = plan[str(size)]
        logfile = output / f"smoke_m{size}.log"
        code = run_stage(stages["smoke"], logfile)
        if code:
            if index + 1 < len(sizes) and "out of memory" in logfile.read_text().lower():
                print(
                    f"Retrying probe microbatch {sizes[index + 1]}; n and R unchanged.",
                    flush=True,
                )
                continue
            raise SystemExit(code)
        events = accept(stages["smoke"])
        report = {
            "accepted": True,
            "local_batch_size": size,
            "events": events,
            "full_run": stages["full"],
            "gpu_validation": "4-rank NCCL SUM reference and actual FSDP2 training smoke",
        }
        (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(run_stage(stages["full"], output / "full.log"))
    raise RuntimeError("No accepted probe microbatch size")


if __name__ == "__main__":
    main()
