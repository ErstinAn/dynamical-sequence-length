#!/usr/bin/env python3
"""Accept packed sequence probes on allocated GPUs, then start the full trajectory."""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_x_monitor_pipeline import run_stage
from scripts.seq_grad_wandb_history import scan_metric_rows


def write_status(output, phase, state, **details):
    record = {"phase": phase, "state": state, "job_id": os.environ.get("SLURM_JOB_ID"),
              "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    temporary = output / "pipeline_status.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output / "pipeline_status.json")
    print(json.dumps(record), flush=True)


def run_check(output, phase, command, env, logfile):
    write_status(output, phase, "running", log=str(logfile))
    with logfile.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_status(output, phase, "failed", exit_code=result.returncode, log=str(logfile))
        raise SystemExit(result.returncode)
    write_status(output, phase, "passed", log=str(logfile))


def accept(stage, pairing_report=None):
    config = yaml.safe_load(Path(stage["config"]).read_text())
    fixed = config["fixed"]
    events = [json.loads(line) for line in
              (Path(stage["dump_folder"]) / "seq_grad_events.jsonl").read_text().splitlines()]
    expected_steps = [0, 2, 4]
    keys = {"context_bias/step", *[f"context_bias/distance_s{s}" for s in fixed["seq_grad.lengths"]]}
    if len(keys) != 17 or [e["metrics"]["context_bias/step"] for e in events] != expected_steps:
        raise RuntimeError("Smoke requires exactly 16 distances at updates 0, 2, 4")
    lengths = [fixed["seq_grad.reference_length"], *fixed["seq_grad.lengths"]]
    for event in events:
        if set(event["metrics"]) != keys or not all(math.isfinite(v) for v in event["metrics"].values()):
            raise RuntimeError("Incorrect or nonfinite sequence payload")
        for length in lengths:
            if not 0 < event["valid_targets"][str(length)] <= fixed["seq_grad.probe_global_batch_size"] * length:
                raise RuntimeError("Invalid global probe target count")
    if fixed.get("seq_grad.sampling", "independent") == "paired_window":
        pairing = json.loads(Path(pairing_report).read_text())
        if not pairing["passed"] or pairing["reference_targets"] != "all_valid_targets_in_full_parent":
            raise RuntimeError("Full-parent/random-window input audit did not pass")
        expected_counts = {row["step"]: row["valid_targets"] for row in pairing["events"]}
        for event in events:
            if event.get("sampling") != "paired_window" or event["valid_targets"] != expected_counts[event["metrics"]["context_bias/step"]]:
                raise RuntimeError("GPU probe targets differ from the audited parent/window batches")
    import wandb

    settings = config["wandb"]
    run = wandb.Api(timeout=60).run(f"{settings['entity']}/{settings['project']}/{stage['wandb_id']}")
    if run.state != "finished":
        raise RuntimeError("Smoke must finish and flush W&B before acceptance")
    for name in ("steps", "seq_len", "global_batch_size", "local_batch_size"):
        if run.config["training"][name] != fixed[f"training.{name}"]:
            raise RuntimeError(f"Uploaded training configuration differs: {name}")
    if run.config["validator"]["eval_seq_len"] != fixed["validator.eval_seq_len"]:
        raise RuntimeError("Uploaded validation length differs from the configured length")
    for key, value in fixed.items():
        if key.startswith("seq_grad.") and run.config["seq_grad_probe"][key.removeprefix("seq_grad.")] != value:
            raise RuntimeError(f"Uploaded probe configuration differs: {key}")
    if run.config["seq_grad_probe"]["dp_degree"] != config["resources"]["gpus_per_node"]:
        raise RuntimeError("Probe DP degree differs from the GPU allocation")
    history = scan_metric_rows(run, sorted(keys))
    if [row["context_bias/step"] for row in history] != expected_steps or any(
        not all(math.isfinite(row[key]) for key in keys) for row in history
    ):
        raise RuntimeError("W&B must receive every smoke event before the full run")
    rows = scan_metric_rows(run, ["n_tokens_seen", "loss_metrics/global_avg_loss", "time_metrics/end_to_end(s)", "lr/AdamW"])
    if len(rows) != 4 or [row["n_tokens_seen"] for row in rows] != [
        step * fixed["training.global_batch_size"] * fixed["training.seq_len"] for step in range(1, 5)
    ] or not all(math.isfinite(row["loss_metrics/global_avg_loss"]) and
                 row["lr/AdamW"] == fixed["optimizer.learning_rate"] for row in rows):
        raise RuntimeError("Training token counts, finite arithmetic or constant LR failed smoke acceptance")
    validation = scan_metric_rows(run, ["validation_metrics/loss", "validation_metrics/evaluated_tokens"])
    if not validation or any(not math.isfinite(row["validation_metrics/loss"]) or
        row["validation_metrics/evaluated_tokens"] != fixed["validator.token_budget"] for row in validation):
        raise RuntimeError("Smoke must finish the configured validation token budget")
    # Exclude compilation at initialization; keep a generous checkpoint/validation margin.
    event_seconds = statistics.median(e["seconds"] for e in events[1:])
    train_seconds = statistics.median(row["time_metrics/end_to_end(s)"] for row in rows[1:])
    return dict(events=events, steady_event_seconds=event_seconds,
                steady_training_step_seconds=train_seconds,
                projected_full_seconds=1.3 * (77 * event_seconds + 3819 * train_seconds) + 1800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = args.plan.parent
    env = dict(os.environ, SEQ_GRAD_TEST_CUDA="1", OMP_NUM_THREADS="1")
    smoke_config = yaml.safe_load(Path(plan["smoke"]["config"]).read_text())
    sampling = smoke_config["fixed"].get("seq_grad.sampling", "independent")
    if sampling == "paired_window":
        run_check(output, "paired_inputs", [sys.executable, "scripts/audit_seq_grad_pairing.py",
                  plan["smoke"]["config"], "--output", str(output / "paired_input_audit.json")],
                  env, output / "paired_input_audit.log")
    commands = [
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
         "-m", "tests.seq_grad_distributed", "--sampling", sampling],
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
         "-m", "tests.seq_grad_microbatch_gpu", "--output", str(output / "state_isolation"),
         "--sampling", sampling],
        [sys.executable, "-m", "tests.packing_attention_gpu", "--output", str(output / "attention.json")],
    ]
    if smoke_config["fixed"].get("seq_grad.parent_population", "packed") == "single_document":
        commands.append([sys.executable, "-m", "tests.seq_grad_single_document_gpu", "--output",
                         str(output / "single_document_attention.json")])
    for index, command in enumerate(commands):
        run_check(output, f"gpu_check_{index}", command, env, output / f"gpu_check_{index}.log")
    write_status(output, "smoke", "running", log=str(output / "smoke.log"))
    code = run_stage(plan["smoke"], output / "smoke.log")
    if code:
        write_status(output, "smoke", "failed", exit_code=code, log=str(output / "smoke.log"))
        raise SystemExit(code)
    try:
        report = accept(plan["smoke"], output / "paired_input_audit.json")
    except Exception as exc:
        write_status(output, "smoke_acceptance", "failed", error=str(exc))
        raise
    report.update(accepted=True, full_run=plan["full"], gpu_validation="4-rank NCCL and FSDP2 state isolation, packed attention")
    end_time = int(os.environ.get("SLURM_JOB_END_TIME", "0"))
    report["remaining_seconds"] = max(0, end_time - time.time()) if end_time else None
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if end_time and report["projected_full_seconds"] > report["remaining_seconds"]:
        write_status(output, "full", "withheld", reason="insufficient_remaining_walltime")
        raise RuntimeError("Acceptance passed, but projected full run exceeds remaining short-QoS allocation; see acceptance.json")
    write_status(output, "full", "running", log=str(output / "full.log"))
    code = run_stage(plan["full"], output / "full.log")
    if code == 0:
        run_check(output, "full_audit", [sys.executable, "scripts/audit_seq_grad_run.py",
                  "--plan", str(args.plan.resolve()), "--stage", "full",
                  "--output", str(output / "full_completion_audit.json")],
                  env, output / "full_audit.log")
    write_status(output, "full", "completed" if code == 0 else "failed", exit_code=code,
                 log=str(output / "full.log"))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
