#!/usr/bin/env python3
"""Verify a finished sequence-distance run against its config, checkpoint and W&B."""

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.seq_grad_wandb_history import scan_metric_rows


LENGTHS = [64, 256, 384, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 2048, 4096]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(config_path, dump_folder, wandb_id):
    import torch.distributed.checkpoint as dcp
    import wandb

    config = yaml.safe_load(config_path.read_text())
    fixed = config["fixed"]
    steps = fixed["training.steps"]
    interval = fixed["seq_grad.interval"]
    expected_events = ([0] if fixed["seq_grad.include_initial"] else []) + list(range(interval, steps + 1, interval))
    require(fixed["seq_grad.lengths"] == LENGTHS, "Candidate lengths differ from the approved protocol")
    require(fixed["seq_grad.reference_length"] == 16384, "Reference length differs from the approved protocol")
    require(fixed["seq_grad.probe_global_batch_size"] == 256, "Global probe batch differs from the approved N=256")
    require((fixed["training.seq_len"], fixed["training.global_batch_size"]) in
            {(1024, 512), (16384, 32)},
            "Expected an approved 1024 / 512 or 16384 / 32 training baseline")
    tokens_per_update = fixed["training.seq_len"] * fixed["training.global_batch_size"]
    validation_manifest = json.loads((Path(fixed["validator.artifact_path"]) / "manifest.json").read_text())
    require(fixed["validator.eval_seq_len"] == validation_manifest["sequence_length"],
            "Validation length must match its fixed artifact")
    require(fixed["optimizer.learning_rate"] == 5e-4 and fixed["lr_scheduler.warmup_steps"] == 0
            and fixed["lr_scheduler.min_lr_factor"] == 1, "Expected constant LR 5e-4")
    keys = {"context_bias/step", *[f"context_bias/distance_s{s}" for s in LENGTHS]}
    events = [json.loads(line) for line in (dump_folder / "seq_grad_events.jsonl").read_text().splitlines()]
    require([e["metrics"]["context_bias/step"] for e in events] == expected_events,
            "Local probe event coordinates are missing, duplicated or incorrect")
    for event in events:
        require(set(event["metrics"]) == keys, "Local probe payload must contain exactly 17 approved keys")
        require(all(math.isfinite(value) and value >= 0 for value in event["metrics"].values()),
                "Local probe has a negative or nonfinite value")
        require(math.isfinite(event["seconds"]) and event["seconds"] > 0
                and math.isfinite(event["peak_allocated_gib"]) and event["peak_allocated_gib"] > 0,
                "Missing real GPU event cost")
        for length in [16384, *LENGTHS]:
            require(0 < event["valid_targets"][str(length)] <= 256 * length,
                    f"Invalid global target count at length {length}")

    checkpoint = dump_folder / "checkpoint" / f"step-{steps}"
    state = {"train_state": {"step": 0, "ntokens_seen": 0, "n_valid_tokens_seen": 0, "seq_grad_last_event": -1,
                             "seq_grad_signature": ""}, "lr_scheduler": {"last_epoch": 0}}
    dcp.load(state, checkpoint_id=checkpoint)
    expected_tokens = steps * tokens_per_update
    require(state["train_state"]["step"] == steps, "Final checkpoint has the wrong optimizer update count")
    # Native Trainer checkpoints a rank-local nominal counter; its W&B metric
    # is the SUM across replicated DP ranks. Valid-target counts are global.
    degree = config["resources"]["gpus_per_node"]
    require(state["train_state"]["ntokens_seen"] * degree == expected_tokens,
            "Checkpoint rank-local token count differs from the training budget")
    require(0 < state["train_state"]["n_valid_tokens_seen"] <= expected_tokens,
            "Checkpoint global valid-target counter is invalid")
    require(state["train_state"]["seq_grad_last_event"] == expected_events[-1], "Checkpoint lost the final scheduled probe")
    require(state["lr_scheduler"]["last_epoch"] == steps, "Scheduler count differs from training updates")

    settings = config["wandb"]
    run_path = f"{settings['entity']}/{settings['project']}/{wandb_id}"
    run = wandb.Api(timeout=60).run(run_path)
    require(run.state == "finished", f"W&B state is {run.state}, not finished")
    for name in ["steps", "seq_len", "global_batch_size", "local_batch_size"]:
        require(run.config["training"][name] == fixed[f"training.{name}"],
                f"W&B training configuration mismatch: {name}")
    require(run.config["validator"]["eval_seq_len"] == fixed["validator.eval_seq_len"],
            "W&B validation length differs from configuration")
    probe = run.config["seq_grad_probe"]
    for key, value in fixed.items():
        if key.startswith("seq_grad."):
            require(probe[key.removeprefix("seq_grad.")] == value, f"W&B config mismatch: {key}")
    require(probe["dp_degree"] == config["resources"]["gpus_per_node"], "W&B reports a different DP degree")
    require(probe["sampler"]["manifest_sha256"] == fixed["dataloader.expected_manifest_sha256"],
            "W&B reports a different dataset population")
    sampling = fixed.get("seq_grad.sampling", "independent")
    protocol = "paired-random-window-v1" if sampling == "paired_window" else "uniform-packed-token-start-v1"
    require(probe["sampler"]["protocol"] == protocol, "Incorrect sampling protocol")
    if sampling == "paired_window":
        single = fixed.get("seq_grad.parent_population", "packed") == "single_document"
        parent_protocol = "uniform-single-document-H-plus-one-window-v1" if single else "uniform-packed-token-start-v1"
        require(probe["sampler"]["parent_protocol"] == parent_protocol
                and probe["sampler"]["reference_targets"] == "all_valid_targets_in_full_parent"
                and probe["sampler"]["reference_length"] == fixed["seq_grad.reference_length"],
                "Paired run must use full-H reference targets from the declared parent population")
        require(all(event.get("sampling") == sampling for event in events),
                "Local events do not identify the paired protocol")
        if single:
            require(probe["sampler"]["minimum_raw_document_tokens"] == fixed["seq_grad.reference_length"] + 1
                    and probe["sampler"]["attention"] == "causal_only", "Incorrect single-document population")
            require(all(event["valid_targets"][str(length)] == 256 * length
                        for event in events for length in [16384, *LENGTHS]),
                    "Single-document probes must use all targets without boundary masking")
    require(probe["learning_rate"] == 5e-4 and probe["lr_scheduler"]["warmup_steps"] == 0
            and probe["lr_scheduler"]["min_lr_factor"] == 1, "W&B metadata does not preserve constant LR")
    require({key for key in run.summary.keys() if key.startswith("context_bias/")} == keys,
            "W&B context metric keys differ from the approved 17 keys")
    history = scan_metric_rows(run, sorted(keys))
    require([row["context_bias/step"] for row in history] == expected_events, "W&B is missing probe events")
    for row, event in zip(history, events, strict=True):
        require(all(math.isfinite(row[key]) and math.isclose(row[key], event["metrics"][key], rel_tol=1e-12, abs_tol=1e-12)
                    for key in keys), "W&B probe values differ from local records")

    training = scan_metric_rows(run, ["n_tokens_seen", "loss_metrics/global_avg_loss", "lr/AdamW"])
    require([row["n_tokens_seen"] for row in training] == [step * tokens_per_update for step in range(1, steps + 1)],
            "W&B training update or token history is incomplete")
    require(all(math.isfinite(row["loss_metrics/global_avg_loss"]) and row["lr/AdamW"] == 5e-4 for row in training),
            "Training losses are nonfinite or learning rate changed")
    validation = scan_metric_rows(run, ["validation_metrics/loss", "validation_metrics/evaluated_tokens"])
    expected_validation = sorted({1, *range(fixed["validator.freq"], steps + 1, fixed["validator.freq"])})
    require([row["_step"] for row in validation] == expected_validation, "Validation history is incomplete")
    require(all(math.isfinite(row["validation_metrics/loss"]) and
                row["validation_metrics/evaluated_tokens"] == fixed["validator.token_budget"] for row in validation),
            "Validation losses or token budgets are incorrect")
    maxima = []
    for length in LENGTHS:
        key = f"context_bias/distance_s{length}"
        event = max(events, key=lambda item: item["metrics"][key])
        maxima.append({"length": length, "maximum_observed_distance": event["metrics"][key],
                       "completed_update": event["metrics"]["context_bias/step"]})
    return {
        "verified_at": datetime.now(timezone.utc).isoformat(), "verified": True,
        "config": str(config_path), "dump_folder": str(dump_folder), "wandb_url": run.url,
        "wandb_state": run.state, "training_updates": steps, "training_tokens": expected_tokens,
        "training_sequence_length": fixed["training.seq_len"],
        "training_global_batch_size": fixed["training.global_batch_size"],
        "first_training_loss": training[0]["loss_metrics/global_avg_loss"],
        "final_training_loss": training[-1]["loss_metrics/global_avg_loss"],
        "constant_lr": 5e-4, "probe_events": len(events), "event_updates": expected_events,
        "probe_metadata": probe, "checkpoint_state": state, "validation": validation,
        "first_event_seconds": events[0]["seconds"],
        "median_steady_event_seconds": statistics.median(e["seconds"] for e in (events[1:] or events)),
        "maximum_peak_allocated_gib": max(e["peak_allocated_gib"] for e in events),
        "observed_maxima": maxima,
        "interpretation": "Maxima over observed finite-batch distances, including sampling noise; not population bounds.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dump-folder", type=Path)
    parser.add_argument("--wandb-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.plan:
        stage = json.loads(args.plan.read_text())[args.stage]
        config, dump, wandb_id = Path(stage["config"]), Path(stage["dump_folder"]), stage["wandb_id"]
    else:
        config, dump, wandb_id = args.config, args.dump_folder, args.wandb_id
        if config is None or dump is None or wandb_id is None:
            parser.error("Provide --plan or all of --config, --dump-folder and --wandb-id")
    report = audit(config, dump, wandb_id)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".maxima.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["observed_maxima"][0]))
        writer.writeheader()
        writer.writerows(report["observed_maxima"])
    print(json.dumps({key: report[key] for key in ["verified", "training_updates", "probe_events", "wandb_url"]}))


if __name__ == "__main__":
    main()
