#!/usr/bin/env python3
"""Plan or submit a Cartesian grid as one collision-safe Slurm job array."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import netrc
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_from_yaml import (
    MANIFEST_VERSION,
    checkpoint_relative_folder,
    config_digest,
    grid,
    latest_complete_step,
    load_config,
    load_manifest,
    manifest_entry,
    resolve_point,
    validate_wandb_id,
    wandb_entity,
)
from storage_paths import runtime_paths, storage_path

from torchtitan_dynamic_seq_len.run_config import source_digest

HERE = Path(__file__).resolve().parent
SLURM_SCRIPT = HERE / "slurm_run_dynamic_seq_len.slurm"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def new_wandb_id() -> str:
    return uuid.uuid4().hex


def submission_environment(run_config_path: Path, manifest_path: Path) -> dict[str, str]:
    """Pass the file paths through the current process environment.

    The command uses --export=ALL.
    An explicit variable list caused login environment retrieval failures in earlier ARC tests."""
    environment = os.environ.copy()
    # The login-node service socket is not available on a compute node.
    environment.pop("WANDB_SERVICE", None)
    environment.update(
        {
            "SLURM_LAUNCHER_DIR": str(HERE),
            "RUN_CONFIG": str(run_config_path),
            "RUN_MANIFEST": str(manifest_path),
        }
    )
    return environment


def run_entry(
    config_path: Path, index: int, wandb_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    _, suffix, parameters, _ = resolve_point(config_path, index)
    dump_folder = (
        Path(config["paths"]["dump_root"]) / str(config["name"]) / validate_wandb_id(wandb_id)
    )
    return {
        "index": index,
        "wandb_id": wandb_id,
        "run_name": f"{config['name']}_{suffix}",
        "suffix": suffix,
        "parameters": parameters,
        "dump_folder": str(dump_folder),
    }


def make_manifest(
    config_path: Path,
    mode: str,
    wandb_ids: list[str],
    *,
    resumed_from: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    _, combos = grid(config)
    if len(wandb_ids) != len(combos):
        raise ValueError("number of W&B IDs must equal the number of sweep grid points")
    return {
        "version": MANIFEST_VERSION,
        "submission_id": uuid.uuid4().hex,
        "mode": mode,
        "created_at": utc_now(),
        "config_path": str(config_path),
        "config_sha256": config_digest(config),
        "source_sha256": source_digest(),
        "experiment": config["name"],
        "torchtitan_config": config["torchtitan_config"],
        "git_commit": git_revision(HERE.parents[1]),
        "torchtitan_commit": git_revision(
            Path(importlib.util.find_spec("torchtitan").origin).parents[1]
        ),
        "resumed_from": str(resumed_from) if resumed_from else None,
        "runs": [
            run_entry(config_path, index, wandb_id, config)
            for index, wandb_id in enumerate(wandb_ids)
        ],
    }


def validate_resume_runs(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    checkpoint_folder = checkpoint_relative_folder(config)
    for run in manifest["runs"]:
        dump_folder = storage_path(run["dump_folder"])
        checkpoint_dir = dump_folder / checkpoint_folder
        step = latest_complete_step(checkpoint_dir)
        if step is None:
            raise ValueError(
                f"resume requested for W&B ID {run['wandb_id']}, but no complete DCP "
                f"checkpoint (.metadata) exists under {checkpoint_dir}"
            )
        run_record_path = dump_folder / "run_manifest.yaml"
        if not run_record_path.is_file():
            raise ValueError(
                f"checkpoint exists for {run['wandb_id']}, but {run_record_path} is "
                "missing; refusing an unverified resume"
            )
        run_record = yaml.safe_load(run_record_path.read_text(encoding="utf-8"))
        if not isinstance(run_record, dict):
            raise ValueError(f"invalid per-run manifest: {run_record_path}")
        if run_record.get("wandb_id") != run["wandb_id"]:
            raise ValueError(f"W&B ID mismatch in {run_record_path}")
        if run_record.get("config_sha256") != config_digest(config):
            raise ValueError(
                f"resume config does not match the original run record {run_record_path}"
            )
        if run_record.get("source_sha256") != source_digest():
            raise ValueError("The source does not match the saved run record.")
        run["resume_step"] = step


def atomic_write_submission(
    root: Path, manifest: dict[str, Any], config: dict[str, Any]
) -> tuple[Path, Path]:
    destination = (
        root / str(manifest["experiment"]) / str(manifest["submission_id"]) / "manifest.yaml"
    )
    destination.parent.mkdir(parents=True, exist_ok=False)
    config_snapshot = destination.parent / "config.yaml"
    config_snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest["config_snapshot"] = str(config_snapshot)
    temporary_manifest = destination.with_suffix(".tmp")
    temporary_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    os.replace(temporary_manifest, destination)
    return destination, config_snapshot


def requested_explicit_id(args: argparse.Namespace, config: dict[str, Any]) -> str | None:
    candidates = [
        ("--wandb-run-id", args.wandb_run_id),
        ("WANDB_RUN_ID", os.environ.get("WANDB_RUN_ID")),
        ("wandb.run_id", (config.get("wandb") or {}).get("run_id")),
    ]
    provided = [(source, str(value).strip()) for source, value in candidates if value]
    if len(provided) > 1 and len({value for _, value in provided}) > 1:
        rendered = ", ".join(f"{source}={value}" for source, value in provided)
        raise ValueError(f"conflicting explicit W&B IDs: {rendered}")
    return validate_wandb_id(provided[0][1]) if provided else None


def wandb_credentials(config_mode: str | None = None) -> tuple[str, str] | None:
    mode = os.environ.get("WANDB_MODE", "") or config_mode or ""
    if mode.lower() in ("offline", "disabled"):
        return ("offline", "")
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if api_key:
        return ("api", api_key)
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    if not host:
        return None
    try:
        credentials = netrc.netrc().authenticators(host)
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    if not credentials or not credentials[2]:
        return None
    return (credentials[0] or "api", credentials[2])


def wandb_auth_available(config_mode: str | None = None) -> bool:
    return wandb_credentials(config_mode) is not None


def verify_wandb_project(team: str, project: str, config_mode: str | None = None) -> None:
    credentials = wandb_credentials(config_mode)
    if credentials is None:
        raise RuntimeError("no W&B credentials are available")
    if credentials[0] == "offline":
        return

    endpoint = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/") + "/graphql"
    payload = json.dumps(
        {
            "query": (
                "query ProjectAccess($entity: String!, $project: String!) { "
                "project(name: $project, entityName: $entity) { name } }"
            ),
            "variables": {"entity": team, "project": project},
        }
    ).encode("utf-8")
    token = base64.b64encode(f"{credentials[0]}:{credentials[1]}".encode("utf-8")).decode("ascii")
    request = Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"W&B project preflight returned HTTP {error.code} for {team}/{project}"
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not verify W&B project {team}/{project}: {error}") from error

    if result.get("errors") or not (result.get("data") or {}).get("project"):
        raise RuntimeError(f"W&B project {team}/{project} is not accessible to the current API key")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show", type=int, metavar="INDEX", help="show one planned grid entry")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--wandb-run-id",
        help="strictly resume a one-point run; also read from WANDB_RUN_ID if omitted",
    )
    resume.add_argument(
        "--resume-manifest",
        type=Path,
        help="strictly resume every run ID in a previous sweep manifest",
    )
    parser.add_argument(
        "--resume-index",
        type=int,
        help="resume only this grid index from --resume-manifest",
    )
    parser.add_argument(
        "--time",
        dest="time_override",
        help="override only the Slurm walltime without changing the training config",
    )
    args = parser.parse_args()

    if args.resume_index is not None and args.resume_manifest is None:
        parser.error("--resume-index requires --resume-manifest")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if "resources" not in config:
        raise ValueError("Add the resources block from cluster/vt_arc/resources.example.yaml.")
    resources = config["resources"]
    paths = runtime_paths(config)

    account = str(resources.get("account", ""))
    if not account or account == "YOUR_ACCOUNT":
        raise ValueError("Set resources.account to your ARC account.")
    partition = str(resources["partition"])
    qos = str(resources["qos"])
    cluster = str(resources["cluster"])
    if not partition or not qos or not cluster:
        raise ValueError("Set resources.cluster, partition, and qos.")
    if int(resources.get("nodes", 1)) != 1:
        raise ValueError("scripts/train.py currently supports exactly one node")
    gpus_per_node = int(resources["gpus_per_node"])
    if gpus_per_node < 1:
        raise ValueError("resources.gpus_per_node must be positive")
    checkpoint_relative_folder(config)

    _, combos = grid(config)
    count = len(combos)
    configured_max_concurrent = int(resources.get("max_concurrent", count))
    if configured_max_concurrent < 1:
        raise ValueError("resources.max_concurrent must be positive")

    wandb_config = config.get("wandb") or {}
    wandb_enabled = bool(wandb_config.get("enabled", False))
    if wandb_enabled and not wandb_auth_available(wandb_config.get("mode")):
        message = (
            "W&B is enabled but no login/API key is available. Run `wandb login` on "
            "the login node, or explicitly set WANDB_MODE=offline."
        )
        if args.dry_run or args.show is not None:
            print(f"WARNING: {message}")
        else:
            raise RuntimeError(message)
    if wandb_enabled and not args.dry_run and args.show is None:
        team = wandb_entity(wandb_config)
        project = str(wandb_config.get("project", "")).strip()
        if not team or not project:
            raise ValueError(
                "online W&B runs require wandb.entity (or legacy wandb.team) and wandb.project"
            )
        verify_wandb_project(team, project, wandb_config.get("mode"))

    explicit_id = None if args.resume_manifest else requested_explicit_id(args, config)
    if args.resume_manifest:
        previous_path = args.resume_manifest.resolve()
        previous = load_manifest(previous_path)
        if previous.get("source_sha256") != source_digest():
            raise ValueError("The source does not match the previous submission.")
        if previous.get("config_sha256") != config_digest(config):
            raise ValueError(
                "resume manifest does not match the current YAML; use the original YAML"
            )
        ids = [
            validate_wandb_id(str(manifest_entry(config_path, previous, index)["wandb_id"]))
            for index in range(count)
        ]
        manifest = make_manifest(config_path, "resume", ids, resumed_from=previous_path)
        if args.resume_index is not None:
            if not 0 <= args.resume_index < count:
                raise IndexError(f"resume index must be in 0..{count - 1}")
            manifest["runs"] = [manifest["runs"][args.resume_index]]
    elif explicit_id:
        if count != 1:
            raise ValueError(
                "one WANDB_RUN_ID cannot identify multiple sweep tasks; use "
                "--resume-manifest for a multi-point sweep"
            )
        manifest = make_manifest(config_path, "resume", [explicit_id])
    else:
        ids: list[str] = []
        used: set[str] = set()
        while len(ids) < count:
            candidate = new_wandb_id()
            dump = Path(paths["dump_root"]) / str(config["name"]) / candidate
            if candidate not in used and not dump.exists():
                used.add(candidate)
                ids.append(candidate)
        manifest = make_manifest(config_path, "new", ids)

    if manifest["mode"] == "resume":
        validate_resume_runs(config, manifest)

    for run in manifest["runs"]:
        resolved_dump = str(storage_path(run["dump_folder"]))
        if resolved_dump != run["dump_folder"]:
            run["runtime_dump_folder"] = resolved_dump

    submission_indices = sorted(int(run["index"]) for run in manifest["runs"])
    max_concurrent = min(configured_max_concurrent, len(submission_indices))
    if submission_indices == list(range(submission_indices[0], submission_indices[-1] + 1)):
        array_range = (
            str(submission_indices[0])
            if len(submission_indices) == 1
            else f"{submission_indices[0]}-{submission_indices[-1]}"
        )
    else:
        array_range = ",".join(map(str, submission_indices))
    requested_time = str(args.time_override or resources["time"]).strip()
    if not requested_time:
        raise ValueError("Slurm walltime must not be empty")
    manifest["slurm_time"] = requested_time

    if args.show is not None:
        entries = [run for run in manifest["runs"] if run["index"] == args.show]
        if len(entries) != 1:
            raise IndexError(f"grid index must be in 0..{count - 1}")
        print(yaml.safe_dump(entries[0], sort_keys=False), end="")
        return

    manifest_root = Path(
        paths.get(
            "manifest_root",
            str(HERE.parents[1] / "outputs/experiments"),
        )
    )
    prospective_manifest = (
        manifest_root
        / str(manifest["experiment"])
        / str(manifest["submission_id"])
        / "manifest.yaml"
    )
    prospective_config = prospective_manifest.parent / "config.yaml"
    if args.dry_run:
        manifest_path = prospective_manifest
        run_config_path = prospective_config
        manifest["config_snapshot"] = str(prospective_config)
        print("DRY RUN: manifest would be written as:")
        print(yaml.safe_dump(manifest, sort_keys=False), end="")
    else:
        try:
            manifest_path, run_config_path = atomic_write_submission(
                manifest_root, manifest, config
            )
        except PermissionError as error:
            raise PermissionError(
                f"cannot write {manifest_root}. Reconnect to ARC so the current shell "
                "has write access to the selected directory."
            ) from error
        print(f"Wrote manifest: {manifest_path}")

    log_root = Path(paths["slurm_log_root"])
    if not args.dry_run:
        log_root.mkdir(parents=True, exist_ok=True)
        for key in ("dump_root", "hf_home", "torch_home"):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    submit_env = submission_environment(run_config_path, manifest_path)
    cmd = [
        "sbatch",
        f"--clusters={cluster}",
        f"--job-name={config['name']}",
        f"--account={account}",
        f"--partition={partition}",
        f"--qos={qos}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{int(resources['gpus_per_node'])}",
        f"--cpus-per-task={int(resources['cpus_per_task'])}",
        f"--time={requested_time}",
        f"--array={array_range}%{max_concurrent}",
        f"--output={log_root}/%x-%A_%a.out",
        f"--error={log_root}/%x-%A_%a.err",
        "--export=ALL",
        str(SLURM_SCRIPT),
    ]
    print(
        f"{'Would submit' if args.dry_run else 'Submitting'} "
        f"{len(submission_indices)} run(s) in "
        f"{manifest['mode']!r} mode, max {max_concurrent} concurrent"
    )
    print(
        "Submission environment: "
        + " ".join(
            f"{name}={shlex.quote(submit_env[name])}"
            for name in ("SLURM_LAUNCHER_DIR", "RUN_CONFIG", "RUN_MANIFEST")
        )
    )
    print(shlex.join(cmd))
    if not args.dry_run:
        raise SystemExit(subprocess.call(cmd, env=submit_env))


if __name__ == "__main__":
    main()
