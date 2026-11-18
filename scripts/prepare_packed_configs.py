"""Prepare six packed runs without replacing historical configurations."""

import copy
import csv
import json
from pathlib import Path

import yaml

from scripts.prepare_longseq_joint_configs import NAMES, compressed_schedule
from scripts.prepare_longseq_loss_grad_configs import make_schedule, STEPS, TARGET
from torchtitan_dynamic_seq_len.data.asset_paths import resolve_archived_asset

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "longseq-finwebedu-packing-llama-100m-seqbs-joint-scheduler"


def main():
    artifact = resolve_archived_asset(ROOT / "assets/data/longseq_validation_llama3_packed_22256")
    manifest = json.loads((artifact / "manifest.json").read_text())
    destination = ROOT / "local/generated_configs/longseq_packed"
    output = ROOT / "outputs/document_packing"
    destination.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for name in NAMES.values():
        source = ROOT / f"local/generated_configs/longseq_loss_grad/{name}.yaml"
        if not source.is_file():
            source = ROOT / f"legacy/configs/longseq_loss_grad/{name}.yaml"
        config = yaml.safe_load(source.read_text())
        config["name"] = "packed-loss22256-grad2048-8192-" + name.replace("_", "-")
        config["torchtitan_config"] = "llama3_100m_packed_documents_loss_grad_seq"
        config["wandb"]["project"] = PROJECT
        config["wandb"]["group"] = "packed-masked-loss22256-grad2048-8192-v1"
        b200 = name in ("bs_increasing_seq_decreasing", "bs_decreasing_seq_increasing")
        config["resources"].update(
            account="jason_optai-paid", cluster="owl" if b200 else "tinkercliffs",
            partition="b200_normal_q" if b200 else "a100_normal_q",
            qos="owl_b200_normal_short" if b200 else "tc_a100_normal_short",
            nodes=1, gpus_per_node=4, cpus_per_task=16, time="24:00:00",
        )
        for key in ("dump_root", "manifest_root", "slurm_log_root", "torch_home"):
            config["paths"][key] = config["paths"][key].replace("longseq_loss_grad", "longseq_packed")
        config["paths"]["validation_dir"] = str(artifact)
        fixed = config["fixed"]
        for key in list(fixed):
            if key.startswith(("grad_seq_len_scheduler.", "batch_size_scheduler.")):
                del fixed[key]
        grad, lengths, batches = make_schedule(name, minimum=2048, maximum=8192)
        fixed.update({"document_masking": True, "dataloader.packing": "stream",
                      "validator.dataloader.packing": "stream",
                      "training.seq_len": 22256, "training.global_batch_size": batches[0],
                      "loss_seq_len_scheduler.length": 22256, "validator.eval_seq_len": 22256,
                      "validator.token_budget": manifest["token_budget"],
                      "validator.eval_local_batch_size": 2,
                      "loss_microbatch_scheduler.batch_size": 2, "loss_tokens_per_microbatch": 44512,
                      "grad_microbatch_scheduler.batch_size": 16, "grad_tokens_per_microbatch": 32768})
        fixed.update({f"grad_seq_len_scheduler.{k}": v for k, v in grad.items()})
        fixed.update({f"batch_size_scheduler.{k}": v for k, v in compressed_schedule(batches).items()})
        path = destination / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        rows = [dict(step=i, parent_length=22256, crop_length=g, global_batch=b,
                     scheduled_gradient_positions=g*b, parent_positions=22256*b)
                for i, (g, b) in enumerate(zip(lengths, batches), 1)]
        with (output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        total = sum(r["scheduled_gradient_positions"] for r in rows)
        report.append(dict(name=name, steps=STEPS, batch_min=min(batches), batch_max=max(batches),
                           scheduled_gradient_positions=total, relative_error=total/TARGET-1,
                           actual_valid_labels="Measured during training; boundary labels do not contribute."))
    # Test both attention settings with the same data and schedule.
    debug_dir = ROOT / "local/generated_configs/longseq_packed_debug"
    debug_dir.mkdir(exist_ok=True)
    for masked in (True, False):
        debug = copy.deepcopy(config)
        suffix = "masked" if masked else "unmasked"
        debug["name"] = "packed-22256-" + suffix + "-smoke"
        debug["wandb"].update(project="dynamic-seq-bs-debug", group="packing-v1-debug")
        debug_artifact = artifact.with_name(artifact.name + "_debug")
        debug["paths"]["validation_dir"] = str(debug_artifact)
        fixed = debug["fixed"]
        fixed["validator.token_budget"] = json.loads((debug_artifact / "manifest.json").read_text())["token_budget"]
        for key in list(fixed):
            if key.startswith(("grad_seq_len_scheduler.", "batch_size_scheduler.")):
                del fixed[key]
        fixed.update({"document_masking": masked, "training.steps": 4, "training.global_batch_size": 10,
                      "batch_size_scheduler.kind": "constant", "batch_size_scheduler.batch_size": 10,
                      "grad_seq_len_scheduler.kind": "piecewise_linear",
                      "grad_seq_len_scheduler.anchor_steps": [1, 2, 3, 4],
                      "grad_seq_len_scheduler.lengths": [2048, 8192, 4096, 8192],
                      "checkpoint.interval": 2, "validator.freq": 4})
        (debug_dir / f"{suffix}.yaml").write_text(yaml.safe_dump(debug, sort_keys=False))
    (output / "schedule_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
