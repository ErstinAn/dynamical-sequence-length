"""Check the completed smoke against W&B records and checkpoint comparison."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

LOSS_KEYS = (
    "loss_metrics/global_avg_loss",
    "loss_metrics/global_max_loss",
    "validation_metrics/loss",
)
LENGTHS = [2048, 2048, 4096, 4096, 8192, 8192, 16384, 16384]
TOKENS = []
for length in LENGTHS:
    TOKENS.append((TOKENS[-1] if TOKENS else 0) + 512 * length)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric_axes(raw_config):
    """Decode W&B's uploaded MetricRecord fields using the installed SDK schema."""
    from wandb.proto import wandb_internal_pb2

    fields = wandb_internal_pb2.MetricRecord.DESCRIPTOR.fields_by_name
    name_field = str(fields["name"].number)
    step_field = str(fields["step_metric"].number)
    index_field = str(fields["step_metric_index"].number)
    definitions = raw_config.get("_wandb", {}).get("m", [])
    require(bool(definitions), "W&B did not return uploaded metric definitions.")
    names = [definition.get(name_field) for definition in definitions]
    axes = {}
    for definition, name in zip(definitions, names):
        if not name:
            continue
        axis = definition.get(step_field)
        index = definition.get(index_field)
        if index is not None:
            require(1 <= int(index) <= len(names), f"Invalid metric axis index for {name}.")
            axis = names[int(index) - 1]
        axes[name] = axis
    return axes


def check_run(api, entity, project, segment, final_step, nproc):
    run_id = segment["wandb_run_id"]
    run = api.run(f"{entity}/{project}/{run_id}")
    require(run.state == "finished", f"W&B run {run_id} is {run.state}.")
    axes = metric_axes(run.rawconfig)
    loss_names = []
    for key in LOSS_KEYS:
        matches = [name for name in axes if name == key or name.endswith("/" + key)]
        require(len(matches) == 1, f"Expected one uploaded definition for {key} in {run_id}.")
        loss_names.append(matches[0])
    prefix = loss_names[0][: -len(LOSS_KEYS[0])]
    require(
        all(name == prefix + key for name, key in zip(loss_names, LOSS_KEYS)),
        f"Loss metric prefixes differ in {run_id}.",
    )
    token_key = prefix + "n_tokens_seen"
    for name in loss_names:
        require(axes[name] == token_key, f"Incorrect token axis for {name} in {run_id}.")
    selected = {
        *loss_names,
        token_key,
        "_step",
        prefix + "seq_len",
        prefix + "global_batch_size",
        prefix + "microbatch_size",
        prefix + "gradient_accumulation_steps",
        prefix + "n_valid_tokens_seen",
        prefix + "valid_tokens_per_step",
        prefix + "validation_metrics/evaluated_tokens",
        prefix + "validation_metrics/num_sequences",
    }
    rows = [
        {key: value for key, value in row.items() if key in selected}
        for row in run.scan_history(page_size=1000)
    ]
    train = [row for row in rows if row.get(loss_names[0]) is not None]
    validation = [row for row in rows if row.get(loss_names[2]) is not None]
    start = int(segment["resume_step"])
    expected = list(range(start + 1, final_step + 1))
    require(len(train) == len(expected), f"Incorrect training row count in {run_id}.")
    losses = {}
    previous_valid = -1
    for step, row in zip(expected, train):
        require(row.get(token_key) == TOKENS[step - 1], f"Incorrect nominal tokens at step {step}.")
        require(row.get("_step") == step, f"Incorrect uploaded optimizer step in {run_id}.")
        require(
            row.get(prefix + "seq_len") == LENGTHS[step - 1],
            f"Incorrect sequence length at step {step}.",
        )
        require(
            row.get(prefix + "global_batch_size") == 512, f"Incorrect global batch at step {step}."
        )
        micro = row.get(prefix + "microbatch_size")
        accumulation = row.get(prefix + "gradient_accumulation_steps")
        require(isinstance(micro, (int, float)) and micro >= 1, "Missing or invalid microbatch.")
        require(
            micro * nproc * accumulation == 512,
            "Microbatch accumulation does not preserve global batch 512.",
        )
        values = [row.get(name) for name in loss_names[:2]]
        require(
            all(isinstance(value, (int, float)) and math.isfinite(value) for value in values),
            f"Missing or nonfinite training loss at step {step}.",
        )
        require(values[0] <= values[1] + 1e-6, f"Average loss exceeds maximum loss at step {step}.")
        valid = row.get(prefix + "n_valid_tokens_seen")
        per_step = row.get(prefix + "valid_tokens_per_step")
        require(
            isinstance(valid, (int, float)) and previous_valid < valid <= TOKENS[step - 1],
            f"Invalid cumulative prediction target count at step {step}.",
        )
        require(
            isinstance(per_step, (int, float)) and 0 < per_step <= 512 * LENGTHS[step - 1],
            f"Invalid prediction target count at step {step}.",
        )
        previous_valid = valid
        losses[step] = values
    # Preserve TorchTitan's native step1-or-frequency validation rule.
    validation_steps = [step for step in expected if step == 1 or step % 8 == 0]
    require(
        len(validation) == len(validation_steps), f"Incorrect validation row count in {run_id}."
    )
    for step, row in zip(validation_steps, validation):
        require(row.get("_step") == step, f"Incorrect validation step in {run_id}.")
        require(
            row.get(token_key) == TOKENS[step - 1],
            "Validation does not use the current cumulative nominal tokens.",
        )
        require(
            row.get(prefix + "validation_metrics/evaluated_tokens") == 15335424,
            "Validation did not evaluate all 15335424 prediction targets.",
        )
        require(
            row.get(prefix + "validation_metrics/num_sequences") == 936,
            "Validation did not use all 936 windows.",
        )
        require(math.isfinite(row[loss_names[2]]), "Validation loss is not finite.")
    return {
        "run_id": run_id,
        "url": run.url,
        "state": run.state,
        "resume_step": start,
        "final_step": final_step,
        "loss_axes": {name: axes[name] for name in loss_names},
        "training_rows": train,
        "validation_rows": validation,
        "loss_by_step": losses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity", default="dynamic_seq_length")
    parser.add_argument("--project", default="debug")
    args = parser.parse_args()
    import wandb

    acceptance = json.loads(args.acceptance.read_text())
    require(acceptance.get("completed_utc"), "The GPU acceptance driver did not complete.")
    require(acceptance.get("token_budget") == 31457280, "Incorrect smoke token budget.")
    comparison = acceptance["checkpoint_comparison"]
    require(comparison.get("exact_match") is True, "Final checkpoints differ.")
    require(
        comparison.get("tensors_checked", 0) > 0
        and comparison.get("non_tensor_values_checked", 0) > 0,
        "Checkpoint comparison did not check tensors and scalar state.",
    )
    api = wandb.Api(timeout=60)
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance": str(args.acceptance.resolve()),
        "token_budget": 31457280,
        "validation_windows": 936,
        "validation_prediction_targets": 15335424,
        "checkpoint_comparison": comparison,
        "cases": {},
    }
    for case in ("uninterrupted", "recovered"):
        directory = Path(acceptance["runs"][case])
        segments = sorted(
            (json.loads(path.read_text()) for path in directory.glob("segment-*.json")),
            key=lambda item: item["resume_step"],
        )
        expected_starts = [0] if case == "uninterrupted" else [0, 4]
        require(
            [item["resume_step"] for item in segments] == expected_starts,
            f"Unexpected resume segments in {case}.",
        )
        records = []
        for index, segment in enumerate(segments):
            end = segments[index + 1]["resume_step"] if index + 1 < len(segments) else 8
            records.append(
                check_run(api, args.entity, args.project, segment, end, acceptance["nproc"])
            )
        result["cases"][case] = records
    reference = result["cases"]["uninterrupted"][0]["loss_by_step"]
    resumed = {
        step: loss
        for segment in result["cases"]["recovered"]
        for step, loss in segment["loss_by_step"].items()
    }
    require(reference == resumed, "Uninterrupted and recovered training losses differ.")
    result["training_losses_match_exactly"] = True
    validation_losses = {}
    for case, segments in result["cases"].items():
        validation_losses[case] = {}
        for segment in segments:
            for row in segment["validation_rows"]:
                keys = [key for key in row if key.endswith("validation_metrics/loss")]
                require(len(keys) == 1, "Expected one validation loss metric.")
                validation_losses[case][row["_step"]] = row[keys[0]]
    require(
        validation_losses["uninterrupted"] == validation_losses["recovered"],
        "Uninterrupted and recovered validation losses differ.",
    )
    result["validation_losses_match_exactly"] = True
    result["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "run_urls": [
                    record["url"] for records in result["cases"].values() for record in records
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
