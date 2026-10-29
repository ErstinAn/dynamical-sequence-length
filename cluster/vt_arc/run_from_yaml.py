#!/usr/bin/env python3
"""Resolve one manifest entry and launch it through the repository's train.py."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from storage_paths import runtime_paths, storage_path

from torchtitan_dynamic_seq_len.run_config import (
    config_digest,
    load_config,
    load_yaml_mapping,
    resolve_point,
    safe_piece,
    source_digest,
    tyro_args,
    wandb_entity,
)
from torchtitan_dynamic_seq_len.run_config import grid as grid

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TRAIN = HERE.parents[1] / "scripts" / "train.py"
MANIFEST_VERSION = 2
WANDB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(config_path: Path, index: int) -> tuple[dict[str, Any], str, list[str]]:
    """Backward-compatible public helper used by existing local tooling."""
    config, suffix, _, args = resolve_point(config_path, index)
    return config, suffix, args


def validate_wandb_id(value: str) -> str:
    value = value.strip()
    if not WANDB_ID_RE.fullmatch(value):
        raise ValueError(f"W&B run ID must match [A-Za-z0-9][A-Za-z0-9_-]{{0,127}}; got {value!r}")
    return value


def checkpoint_relative_folder(config: dict[str, Any]) -> Path:
    fixed = config.get("fixed") or {}
    if fixed.get("checkpoint.enable") is not True:
        raise ValueError("fixed.checkpoint.enable must be true for resumable runs")
    folder = Path(str(fixed.get("checkpoint.folder", "checkpoint")))
    if folder.is_absolute() or ".." in folder.parts:
        raise ValueError("fixed.checkpoint.folder must be a safe relative path")
    return folder


def latest_complete_step(checkpoint_dir: Path) -> int | None:
    if not checkpoint_dir.is_dir():
        return None
    steps: list[int] = []
    for entry in checkpoint_dir.iterdir():
        match = re.fullmatch(r"step-(\d+)", entry.name)
        if match and entry.is_dir() and (entry / ".metadata").is_file():
            steps.append(int(match.group(1)))
    return max(steps) if steps else None


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_yaml_mapping(path, "run manifest")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest version {manifest.get('version')!r}; expected {MANIFEST_VERSION}"
        )
    if manifest.get("mode") not in ("new", "resume"):
        raise ValueError("manifest.mode must be 'new' or 'resume'")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest.runs must be a non-empty list")
    return manifest


def manifest_entry(config_path: Path, manifest: dict[str, Any], index: int) -> dict[str, Any]:
    config = load_config(config_path)
    if manifest.get("config_sha256") != config_digest(config):
        raise ValueError(
            "run manifest does not match the current YAML. Resume with the original "
            "YAML, or start a new run after changing training parameters."
        )
    entries = [run for run in manifest["runs"] if run.get("index") == index]
    if len(entries) != 1:
        raise ValueError(f"manifest must contain exactly one entry for index {index}")
    entry = entries[0]
    validate_wandb_id(str(entry.get("wandb_id", "")))

    _, expected_suffix, expected_parameters, _ = resolve_point(config_path, index)
    if entry.get("suffix") != expected_suffix or entry.get("parameters") != expected_parameters:
        raise ValueError(
            f"manifest entry {index} does not match the YAML sweep grid: "
            f"suffix={entry.get('suffix')!r} expected={expected_suffix!r}, "
            f"parameters={entry.get('parameters')!r} expected={expected_parameters!r}"
        )
    expected_dump = (
        Path(config["paths"]["dump_root"]) / str(config["name"]) / str(entry["wandb_id"])
    )
    if Path(entry.get("dump_folder", "")) != expected_dump:
        raise ValueError(
            f"manifest dump_folder must be {expected_dump}, got {entry.get('dump_folder')!r}"
        )
    return entry


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def writer_lock(dump_folder: Path, entry: dict[str, Any]) -> Iterator[None]:
    lock = dump_folder / ".writer-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        details = lock / "owner.yaml"
        owner = details.read_text(encoding="utf-8").strip() if details.is_file() else ""
        raise RuntimeError(
            f"checkpoint directory already has an active/stale writer lock: {lock}\n"
            f"{owner}\nVerify the recorded Slurm job is no longer running before removing it."
        ) from error

    owner = {
        "wandb_id": entry["wandb_id"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": utc_now(),
    }
    atomic_write_yaml(lock / "owner.yaml", owner)
    try:
        yield
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    config, _, _, passthrough = resolve_point(config_path, args.index)
    manifest = load_manifest(manifest_path)
    entry = manifest_entry(config_path, manifest, args.index)
    if manifest.get("source_sha256") != source_digest():
        raise ValueError("The source files changed after submission. Restore the submitted source.")
    resources = config["resources"]
    paths = runtime_paths(config)
    gpus = int(resources["gpus_per_node"])

    wandb_id = str(entry["wandb_id"])
    run_name = str(entry["run_name"])
    dump_folder = storage_path(entry["dump_folder"])
    checkpoint_dir = dump_folder / checkpoint_relative_folder(config)
    mode = str(manifest["mode"])
    latest_step = latest_complete_step(checkpoint_dir)
    if mode == "resume" and latest_step is None:
        raise RuntimeError(
            f"resume requested for W&B ID {wandb_id}, but no complete TorchTitan DCP "
            f"checkpoint was found in {checkpoint_dir}"
        )
    if mode == "new" and latest_step is not None:
        raise RuntimeError(
            f"new run {wandb_id} unexpectedly points at an existing checkpoint in "
            f"{checkpoint_dir}; refusing to overwrite it"
        )

    os.environ["HF_HOME"] = str(paths["hf_home"])
    os.environ["HF_DATASETS_CACHE"] = str(Path(paths["hf_home"]) / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(paths["hf_home"]) / "hub")
    os.environ["HF_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]
    os.environ["TORCH_HOME"] = str(paths["torch_home"])
    if paths.get("validation_dir"):
        # Training exclusions and fixed validation must resolve the same artifact.
        os.environ["DYNAMIC_SEQ_LEN_VALIDATION_DIR"] = paths["validation_dir"]
    # Separate compiled kernels across TinkerCliffs/Owl GPU environments.
    cluster = safe_piece(os.environ.get("SLURM_CLUSTER_NAME", socket.gethostname()))
    compile_root = Path(paths["torch_home"]) / "compile_cache" / cluster
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_root / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(compile_root / "triton")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    wandb = config.get("wandb") or {}
    enabled = bool(wandb.get("enabled", False))
    if enabled:
        os.environ["WANDB_PROJECT"] = str(wandb.get("project", "torchtitan"))
        entity = wandb_entity(wandb)
        if entity:
            os.environ["WANDB_TEAM"] = entity  # TorchTitan's WandBLogger reads this.
            os.environ["WANDB_ENTITY"] = entity
        os.environ["WANDB_RUN_GROUP"] = str(wandb.get("group", config["name"]))
        os.environ["WANDB_RUN_NAME"] = run_name
        os.environ["WANDB_RUN_ID"] = wandb_id
        os.environ["WANDB_RESUME"] = "must" if mode == "resume" else "never"
        if wandb.get("mode") and "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = str(wandb["mode"])
    passthrough.extend(tyro_args("metrics.enable_wandb", enabled))
    passthrough.extend(["--dump_folder", str(dump_folder)])

    cmd = [
        sys.executable,
        str(TRAIN),
        str(config["torchtitan_config"]),
        "--nproc",
        str(gpus),
        *passthrough,
    ]
    print(f"[slurm-run] mode={mode} index={args.index} run={run_name} wandb_id={wandb_id}")
    print(f"[slurm-run] dump_folder={dump_folder}")
    print(
        "[slurm-run] "
        + " ".join(
            f"{key}={os.environ.get(key, '<unset>')!r}"
            for key in (
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
                "SLURM_TRES_BIND",
                "CUDA_VISIBLE_DEVICES",
            )
        ),
        flush=True,
    )
    if str(dump_folder) != entry["dump_folder"]:
        print(f"[slurm-run] relocated_from={entry['dump_folder']}")
    if latest_step is not None:
        print(f"[slurm-run] latest_complete_checkpoint=step-{latest_step}")
    print(f"[slurm-run] {shlex.join(cmd)}", flush=True)
    if args.dry_run:
        return

    dump_folder.mkdir(parents=True, exist_ok=True)
    with writer_lock(dump_folder, entry):
        run_record = {
            **entry,
            "source_dump_folder": entry["dump_folder"],
            "dump_folder": str(dump_folder),
            "manifest": str(manifest_path),
            "mode": mode,
            "config_sha256": manifest["config_sha256"],
            "source_sha256": manifest["source_sha256"],
            "torchtitan_config": config["torchtitan_config"],
            "launched_at": utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        atomic_write_yaml(dump_folder / "run_manifest.yaml", run_record)
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
