#!/usr/bin/env python3
"""Start one experiment on one node."""

import argparse
import ast
import fcntl
import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN_BUILDER = "llama3_1b"
MAIN_MODULE = "torchtitan_dynamic_seq_len.experiment"


def builders():
    tree = ast.parse((ROOT / "torchtitan_dynamic_seq_len/config_registry.py").read_text())
    return [MAIN_BUILDER] + [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
        and node.returns is not None
    ]


def main_preflight(config, arguments, nproc):
    """Validate the complete main experiment without allocating a GPU."""
    from torchtitan.config import ConfigManager
    from torchtitan_dynamic_seq_len.run_config import wandb_entity

    required = ("token_budget", "training.steps", "training.seq_len",
                "training.global_batch_size", "training.local_batch_size",
                "seq_len_scheduler.kind", "batch_size_scheduler.kind",
                "microbatch_scheduler.kind", "optimizer.learning_rate",
                "lr_scheduler.warmup_steps", "lr_scheduler.decay_type",
                "lr_scheduler.min_lr_factor")
    explicit = set(config.get("fixed", {})) | set(config.get("sweep", {}))
    missing = sorted(set(required) - explicit)
    if not ({"validator.freq", "validator.token_intervals"} & explicit):
        missing.append("validator.freq or validator.token_intervals")
    if missing:
        raise ValueError("Specify experiment settings explicitly: " + ", ".join(missing))
    wandb = config.get("wandb", {})
    if (wandb.get("enabled") is not True or wandb.get("mode", "online") != "online"
            or not wandb_entity(wandb) or not wandb.get("project")):
        raise ValueError("The main experiment requires online W&B with an entity and project.")
    previous_world = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(nproc)
    try:
        parsed = ConfigManager().parse_args(
            ["--module", MAIN_MODULE, "--config", MAIN_BUILDER, *arguments])
        parsed.validate_dynamic()
        if not parsed.validator.token_intervals and (
                "validator.freq" not in explicit or parsed.validator.freq < 1):
            raise ValueError("Step-based validation requires a positive explicit validator.freq.")
    finally:
        if previous_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = previous_world
    if parsed.parallelism.data_parallel_replicate_degree != nproc:
        raise ValueError("DP degree must match --nproc for this single-node launcher.")
    return parsed


def latest_complete_step(folder):
    steps = []
    for path in folder.glob("step-*"):
        suffix = path.name.removeprefix("step-")
        if suffix.isdigit() and (path / ".metadata").is_file():
            steps.append(int(suffix))
    if not steps:
        raise ValueError(f"No complete checkpoint in {folder}")
    return max(steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/smoke.yaml")
    parser.add_argument("--nproc", type=int, help="Override the GPU count from the YAML for a new run.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-legacy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-dir", type=Path, help="Resume a main experiment from its run directory.")
    parser.add_argument("--stop-after-checkpoint", type=int, choices=(2, 4, 6),
                        help="Stop an eight-step smoke after a complete checkpoint for recovery checks.")
    args, extra = parser.parse_known_args()
    if args.list:
        print(MAIN_BUILDER)
        return
    if args.list_legacy:
        print("\n".join(name for name in builders() if name != MAIN_BUILDER))
        return
    resume = None
    if args.resume_dir:
        if extra or args.config != "configs/smoke.yaml" or args.index:
            parser.error("Resume uses the recorded configuration and arguments. Do not supply overrides.")
        args.resume_dir = args.resume_dir.resolve()
        try:
            resume = json.loads((args.resume_dir / "launch.json").read_text())
        except (OSError, ValueError) as error:
            parser.error(str(error))
        args.config = str(args.resume_dir / "experiment.yaml")
        if args.nproc is None:
            args.nproc = resume["nproc"]
    if args.nproc is not None and args.nproc < 1:
        parser.error("Set --nproc to a positive integer.")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.pop("DYNAMIC_SEQ_LEN_SMOKE_STOP_AFTER", None)
    config_path = Path(args.config)
    config = None
    dump = None
    if config_path.suffix in (".yaml", ".yml"):
        from torchtitan_dynamic_seq_len.run_config import (
            config_digest, resolve_point, source_digest, tyro_args, wandb_entity,
        )

        try:
            config, _, _, fixed = resolve_point(config_path, args.index)
        except (ValueError, FileNotFoundError, IndexError) as error:
            parser.error(str(error))
        builder = config["torchtitan_config"]
        paths = config["paths"]
        if paths.get("validation_dir"):
            env["DYNAMIC_SEQ_LEN_VALIDATION_DIR"] = paths["validation_dir"]
        if paths.get("hf_home"):
            env["HF_HOME"] = paths["hf_home"]
            env["HF_DATASETS_CACHE"] = str(Path(paths["hf_home"]) / "datasets")
            env["HF_HUB_CACHE"] = str(Path(paths["hf_home"]) / "hub")
            env["HUGGINGFACE_HUB_CACHE"] = env["HF_HUB_CACHE"]
        if paths.get("torch_home"):
            env["TORCH_HOME"] = paths["torch_home"]
        run_id = uuid.uuid4().hex
        dump = Path(paths["dump_root"]) / config["name"] / run_id
        experiment_id = run_id
        if resume:
            if (builder != MAIN_BUILDER or resume["config_sha256"] != config_digest(config)
                    or resume["source_sha256"] != source_digest() or resume["nproc"] != args.nproc):
                parser.error("Resume requires the recorded source, configuration, and DP degree.")
            dump = args.resume_dir
            experiment_id = resume["experiment_id"]
        wandb = config.get("wandb") or {}
        enabled = bool(wandb.get("enabled", False))
        if enabled:
            env.update(
                WANDB_PROJECT=str(wandb["project"]),
                WANDB_RUN_ID=run_id,
                WANDB_RUN_NAME=config["name"],
                WANDB_RESUME="never",
                WANDB_MODE=str(wandb.get("mode", "online")),
            )
            if builder == MAIN_BUILDER:
                env["WANDB_RUN_GROUP"] = experiment_id
                for key in ("WANDB_RESUME_FROM", "WANDB_FORK_FROM", "WANDB_SERVICE"):
                    env.pop(key, None)
            entity = wandb_entity(wandb)
            if entity:
                env.update(WANDB_ENTITY=entity, WANDB_TEAM=entity)
        fixed += tyro_args("metrics.enable_wandb", enabled)
        fixed += ["--dump_folder", str(dump)]
        extra = fixed + extra
        if resume:
            extra = resume["arguments"]
    else:
        builder = args.config
    if builder not in builders():
        parser.error("Unknown configuration. Use --list to show the names.")
    if args.nproc is None:
        args.nproc = (config or {}).get("num_gpus", 1)
    module = MAIN_MODULE if builder == MAIN_BUILDER else "torchtitan_dynamic_seq_len.config_registry"
    plan = None
    if builder == MAIN_BUILDER:
        if config is None:
            parser.error("Use a YAML file with explicit experiment settings for the main entry.")
        try:
            parsed = main_preflight(config, extra, args.nproc)
            if args.stop_after_checkpoint:
                if not parsed.smoke_test:
                    raise ValueError("--stop-after-checkpoint is only available for smoke_test.")
                env["DYNAMIC_SEQ_LEN_SMOKE_STOP_AFTER"] = str(args.stop_after_checkpoint)
            plan = parsed.build_token_plan().to_dict()
            if resume:
                checkpoint_root = Path(dump) / parsed.checkpoint.folder
                resume_step = latest_complete_step(checkpoint_root)
                if resume_step >= parsed.training.steps:
                    raise ValueError("This run already has its final checkpoint.")
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(plan, indent=2), flush=True)
    elif args.stop_after_checkpoint:
        parser.error("--stop-after-checkpoint requires the main smoke entry.")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        "--local-ranks-filter=0",
        "--role=rank",
        "--tee=3",
        "-m",
        "torchtitan.train",
        "--module",
        module,
        "--config",
        builder,
        *extra,
    ]
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    import torch

    if torch.cuda.device_count() < args.nproc:
        parser.error("The allocation does not contain enough visible GPUs.")
    if config is not None:
        import yaml

        dump.mkdir(parents=True, exist_ok=bool(resume))
        with (dump / ".writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("Another process holds this run directory.")
            if not resume:
                (dump / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
                (dump / "command.txt").write_text(shlex.join(cmd) + "\n")
            if plan:
                if not resume:
                    record = dict(experiment_id=experiment_id, nproc=args.nproc,
                                  arguments=extra, config_sha256=config_digest(config),
                                  source_sha256=source_digest())
                    (dump / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
                    (dump / "token_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
                segment = dict(wandb_run_id=run_id, experiment_id=experiment_id,
                               resume_step=resume_step if resume else 0, command=cmd)
                (dump / f"segment-{run_id}.json").write_text(json.dumps(segment, indent=2) + "\n")
            raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))
    raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))


if __name__ == "__main__":
    main()
