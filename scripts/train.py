#!/usr/bin/env python3
"""Start one experiment on one node."""

import argparse
import ast
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def builders():
    tree = ast.parse((ROOT / "torchtitan_dynamic_seq_len/config_registry.py").read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
        and node.returns is not None
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/smoke.yaml")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if args.list:
        print("\n".join(builders()))
        return
    if args.nproc < 1:
        parser.error("Set --nproc to a positive integer.")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    config_path = Path(args.config)
    config = None
    dump = None
    if config_path.suffix in (".yaml", ".yml"):
        from torchtitan_dynamic_seq_len.run_config import resolve_point, tyro_args, wandb_entity

        config, _, _, fixed = resolve_point(config_path, args.index)
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
            entity = wandb_entity(wandb)
            if entity:
                env.update(WANDB_ENTITY=entity, WANDB_TEAM=entity)
        fixed += tyro_args("metrics.enable_wandb", enabled)
        fixed += ["--dump_folder", str(dump)]
        extra = fixed + extra
    else:
        builder = args.config
    if builder not in builders():
        parser.error("Unknown configuration. Use --list to show the names.")
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
        "torchtitan_dynamic_seq_len.config_registry",
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

        dump.mkdir(parents=True, exist_ok=False)
        (dump / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (dump / "command.txt").write_text(shlex.join(cmd) + "\n")
    raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))


if __name__ == "__main__":
    main()
