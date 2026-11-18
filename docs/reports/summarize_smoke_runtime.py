"""Summarize completed smoke timings and checkpoint storage from local records."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    steps = {}
    logs = sorted((directory / "structured_logs").glob("training.global_rank_0.*.jsonl"))
    for logfile in logs:
        for line in logfile.read_text().splitlines():
            row = json.loads(line)
            step = row.get("step")
            event = row.get("log_type_name")
            if step is None or event not in {
                "step_start",
                "step_end",
                "checkpoint_save_start",
                "checkpoint_save_end",
            }:
                continue
            events = steps.setdefault(int(step), {})
            if event in events:
                raise ValueError(f"Duplicate {event} for step {step}.")
            events[event] = row
    if set(steps) != set(range(1, 9)):
        raise ValueError("Expected all eight complete optimizer steps.")
    lengths = (2048, 2048, 4096, 4096, 8192, 8192, 16384, 16384)
    timings = []
    for step, events in sorted(steps.items()):
        begin = events["step_start"]["time_us"]
        save_begin = events["checkpoint_save_start"]["time_us"]
        save_end = events["checkpoint_save_end"]["time_us"]
        end = events["step_end"]["time_us"]
        training = (save_begin - begin) / 1e6
        nominal = 512 * lengths[step - 1]
        timings.append(
            {
                "step": step,
                "sequence_length": lengths[step - 1],
                "nominal_tokens": nominal,
                "training_and_gc_seconds": training,
                "checkpoint_call_seconds": events["checkpoint_save_end"]["value"] / 1000,
                "validation_and_loop_overhead_seconds": (end - save_end) / 1e6,
                "whole_step_seconds": events["step_end"]["value"] / 1000,
                "nominal_tokens_per_training_second": nominal / training,
                "saved_checkpoint": step in (2, 4, 6, 8),
                "ran_validation": step in (1, 8),
            }
        )
    checkpoints = []
    for checkpoint in sorted((directory / "checkpoint").glob("step-*")):
        if not (checkpoint / ".metadata").is_file():
            raise ValueError(f"Incomplete checkpoint: {checkpoint}")
        files = [path for path in checkpoint.rglob("*") if path.is_file()]
        checkpoints.append(
            {
                "path": str(checkpoint),
                "files": len(files),
                "apparent_bytes": sum(path.stat().st_size for path in files),
                "allocated_bytes": sum(path.stat().st_blocks * 512 for path in files),
            }
        )
    if len(checkpoints) != 4:
        raise ValueError("Expected four complete smoke checkpoints per case.")
    return {
        "run_directory": str(directory),
        "rank0_logs": [str(path) for path in logs],
        "step_timings": timings,
        "checkpoint_storage": checkpoints,
        "all_checkpoint_apparent_bytes": sum(item["apparent_bytes"] for item in checkpoints),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    if not acceptance.get("completed_utc"):
        raise ValueError("Wait for the acceptance driver to complete.")
    result = {
        "timing_method": {
            "rank": 0,
            "training": "step_start to checkpoint_save_start, including garbage collection",
            "checkpoint": "checkpoint_save_end span value, milliseconds converted to seconds",
            "validation": "checkpoint_save_end to step_end, including loop/profiler overhead",
            "limitation": "Eight-step measurements on two H200 GPUs. Do not extrapolate to another GPU count without a new measurement.",
        },
        "cases": {name: summarize(Path(path)) for name, path in acceptance["runs"].items()},
    }
    maximum = max(
        checkpoint["apparent_bytes"]
        for case in result["cases"].values()
        for checkpoint in case["checkpoint_storage"]
    )
    result["largest_checkpoint_bytes"] = maximum
    result["ten_checkpoint_storage_estimate_bytes"] = maximum * 10
    result["ten_checkpoint_estimate_scope"] = (
        "Same model, optimizer, and checkpoint state. Add space for temporary writes and other files."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "largest_checkpoint_bytes": maximum,
                "ten_checkpoint_storage_estimate_bytes": maximum * 10,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
