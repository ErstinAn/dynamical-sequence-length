"""Prepare six loss/crop experiments with matched gradient token budgets."""

import copy
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_longseq_joint_configs import NAMES, compressed_schedule
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig

STEPS = 3840
TARGET = 12079595520
PERIOD = 320


def values(config):
    schedule = SeqLenSchedulerConfig(**config).build(STEPS)
    return [schedule.length_for_step(step) for step in range(1, STEPS + 1)]


def match_budget(lengths, weights):
    """Scale the batch trend, then round each global batch to eight documents."""
    def candidate(scale):
        batches = [max(8, 8 * int(scale * weight / 8 + 0.5)) for weight in weights]
        return sum(g * b for g, b in zip(lengths, batches)), batches

    scale = TARGET / sum(g * w for g, w in zip(lengths, weights))
    low, high = scale / 2, scale * 2
    for _ in range(60):
        middle = (low + high) / 2
        if candidate(middle)[0] < TARGET:
            low = middle
        else:
            high = middle
    total, batches = min((candidate(low), candidate(high)), key=lambda x: abs(x[0] - TARGET))
    assert abs(total / TARGET - 1) < 0.0004
    return batches


def make_schedule(name, minimum=1024, maximum=12288):
    if name in ("bs_increasing_seq_decreasing", "bs_decreasing_seq_increasing"):
        endpoints = [minimum, maximum]
        if name == "bs_increasing_seq_decreasing":
            endpoints.reverse()
        grad = dict(kind="piecewise_linear", anchor_steps=[1, STEPS],
                    lengths=endpoints, round_to_multiple=8)
        lengths = values(grad)
        weights = [1 / g for g in lengths]
    elif name == "bs_cyclic_seq_inverse":
        # Start at the maximum crop, as in the reference inverse batch cycle.
        # A compact piecewise triangle includes one sample at each extremum.
        grad = dict(kind="piecewise_linear", anchor_steps=[1, 161, 320],
                    lengths=[maximum, minimum, (maximum - 8 * round((maximum - minimum) / 160 / 8))], round_to_multiple=8, repeat=True)
        lengths = values(grad)
        weights = [1 / g for g in lengths]
    elif name.startswith("linear_bs_cyclic_seq_"):
        # Equal endpoint samples make each cycle symmetric. This removes drift
        # in the integrated token budget when the batch grows across cycles.
        grad = dict(kind="piecewise_linear", anchor_steps=[1, 160, 161, 320],
                    lengths=[minimum, maximum, maximum, minimum], round_to_multiple=8, repeat=True)
        lengths = values(grad)
        weights = [1 + 2 * i / (STEPS - 1) for i in range(STEPS)]
    elif name == "four_stage_cycle":
        anchors = [1, 80, 81, 160, 161, 240, 241, 320]
        grad = dict(kind="piecewise_linear", anchor_steps=anchors,
                    lengths=[minimum, minimum, minimum, minimum, maximum, maximum, minimum, minimum],
                    round_to_multiple=8, repeat=True)
        lengths = values(grad)
        weights = [b / 1000 for b in values(dict(
            kind="piecewise_linear", anchor_steps=anchors,
            lengths=[1000, 1000, 1000, 4000, 4000, 4000, 1000, 1000], repeat=True))]
    else:
        raise ValueError(name)
    return grad, lengths, match_budget(lengths, weights)


def main():
    destination = ROOT / "local/generated_configs/longseq_loss_grad"
    output = ROOT / "outputs/longseq_loss_grad"
    destination.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=True)
    records = json.loads((ROOT / "outputs/longseq_joint/finweb-llama-100m-seqbs-joint-scheduler.json").read_text())
    report = []
    for record in records:
        name = NAMES[record["id"]]
        source = ROOT / f"local/generated_configs/longseq_joint/{name}.yaml"
        if not source.is_file():
            source = ROOT / f"legacy/configs/longseq_joint/{name}.yaml"
        config = yaml.safe_load(source.read_text())
        config["name"] = "longseq-loss12288-103m-" + name.replace("_", "-") + "-3840steps"
        config["torchtitan_config"] = "llama3_100m_long_documents_loss_grad_seq"
        for key in ("dump_root", "manifest_root", "slurm_log_root", "torch_home"):
            config["paths"][key] = config["paths"][key].replace("longseq_joint", "longseq_loss_grad")
        config["resources"].update(cluster="tinkercliffs", partition="a100_normal_q",
                                    qos="tc_a100_normal_short", gpus_per_node=4,
                                    mem="64G", time="24:00:00")
        config["wandb"]["group"] = "loss12288-grad1024-12288-3840steps-v2"
        fixed = config["fixed"]
        for key in list(fixed):
            if key.startswith(("seq_len_scheduler.", "microbatch_scheduler.", "batch_size_scheduler.")):
                del fixed[key]
        grad, lengths, batches = make_schedule(name)
        fixed.update({
            "training.steps": STEPS, "training.local_batch_size": 1,
            "training.global_batch_size": batches[0], "loss.num_chunks": 8,
            "loss_seq_len_scheduler.kind": "constant", "loss_seq_len_scheduler.length": 12288,
            "loss_microbatch_scheduler.kind": "constant", "loss_microbatch_scheduler.batch_size": 8,
            "grad_microbatch_scheduler.kind": "constant", "grad_microbatch_scheduler.batch_size": 48,
            "loss_tokens_per_microbatch": 98304, "grad_tokens_per_microbatch": 49152,
            "crop_seed": 1, "checkpoint.interval": 100,
        })
        fixed.update({f"grad_seq_len_scheduler.{k}": v for k, v in grad.items()})
        fixed.update({f"batch_size_scheduler.{k}": v for k, v in compressed_schedule(batches).items()})
        path = destination / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        rows = [dict(step=s, loss_seq_len=12288, grad_seq_len=g, global_batch=b,
                     grad_tokens=g * b, parent_tokens=12288 * b)
                for s, (g, b) in enumerate(zip(lengths, batches), 1)]
        with (output / f"{name}_schedule.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        tokens = sum(row["grad_tokens"] for row in rows)
        report.append(dict(name=name, config=str(path), reference_url=record["url"],
                           steps=STEPS, grad_min=min(lengths), grad_max=max(lengths),
                           batch_min=min(batches), batch_max=max(batches),
                           grad_tokens=tokens, target_grad_tokens=TARGET,
                           relative_budget_error=tokens / TARGET - 1,
                           parent_tokens=sum(row["parent_tokens"] for row in rows),
                           cycles=12 if grad.get("repeat") else None))
    # Real data, full vocabulary, short-crop extreme, maximum crop, and odd shapes.
    debug = copy.deepcopy(config)
    debug["name"] = "longseq-loss-grad-native-vocab-smoke"
    debug["wandb"].update(project="dynamic-seq-bs-debug", group="longseq-loss-grad-v2-debug")
    fixed = debug["fixed"]
    for key in list(fixed):
        if key.startswith(("grad_seq_len_scheduler.", "batch_size_scheduler.")):
            del fixed[key]
    fixed.update({"training.steps": 8, "training.global_batch_size": 32,
                  "batch_size_scheduler.kind": "constant", "batch_size_scheduler.batch_size": 32,
                  "grad_seq_len_scheduler.kind": "piecewise_linear",
                  "grad_seq_len_scheduler.anchor_steps": list(range(1, 9)),
                  "grad_seq_len_scheduler.lengths": [1024, 1096, 6656, 12288, 12216, 1168, 1024, 12288],
                  "grad_seq_len_scheduler.round_to_multiple": 8,
                  "checkpoint.interval": 4, "validator.freq": 8})
    debug_path = ROOT / "local/generated_configs/longseq_loss_grad_debug/shapes.yaml"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(yaml.safe_dump(debug, sort_keys=False))
    (output / "schedule_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
