"""Read experiment settings and select one point from a sweep."""

import hashlib
import itertools
import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{description} {path} must contain a YAML mapping")
    return data


def load_config(path: Path) -> dict[str, Any]:
    data = load_yaml_mapping(path, "config")
    if "num_gpus" in data and (type(data["num_gpus"]) is not int or data["num_gpus"] < 1):
        raise ValueError(f"{path}: num_gpus must be a positive integer")
    if data.get("status") == "draft":
        raise ValueError(
            f"{path}: this configuration is a placeholder. "
            "Complete the experiment settings and server launch example described in "
            "docs/codebase_kick_off/getting_started.md before removing status: draft."
        )
    for required in ("name", "torchtitan_config", "paths"):
        if required not in data:
            raise ValueError(f"{path}: missing required key {required!r}")
    roots = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "ASSETS_DIR": os.environ.get("DYNAMIC_SEQ_LEN_ASSETS", str(PROJECT_ROOT / "assets")),
        "OUTPUT_ROOT": os.environ.get("DYNAMIC_SEQ_LEN_OUTPUT", str(PROJECT_ROOT / "outputs")),
    }

    def expand(value):
        if isinstance(value, str):
            for name, root in roots.items():
                value = value.replace("${" + name + "}", root)
            return value
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    wandb_entity(data.get("wandb") or {})
    return expand(data)


def wandb_entity(config: dict[str, Any]) -> str:
    """Use W&B's entity name; retain `team` for existing experiment snapshots."""
    entity = str(config.get("entity") or "").strip()
    team = str(config.get("team") or "").strip()
    if entity and team and entity != team:
        raise ValueError("wandb.entity and legacy wandb.team disagree")
    return entity or team


def config_digest(config: dict[str, Any]) -> str:
    rendered = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def grid(config: dict[str, Any]) -> tuple[list[str], list[tuple[Any, ...]]]:
    sweep = config.get("sweep") or {}
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a mapping from dotted CLI keys to value lists")
    keys = list(sweep)
    values: list[list[Any]] = []
    for key in keys:
        choices = sweep[key]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"sweep.{key} must be a non-empty list")
        values.append(choices)
    return keys, list(itertools.product(*values)) if values else [()]


def tyro_args(key: str, value: Any) -> list[str]:
    if isinstance(value, bool):
        if value:
            return [f"--{key}"]
        parent, _, leaf = key.rpartition(".")
        return [f"--{parent}.no-{leaf}" if parent else f"--no-{leaf}"]
    if isinstance(value, list):
        return [f"--{key}", *map(str, value)]
    if value is None:
        return []
    return [f"--{key}", str(value)]


def safe_piece(value: Any) -> str:
    text = "-".join(map(str, value)) if isinstance(value, list) else str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def resolve_point(
    config_path: Path, index: int
) -> tuple[dict[str, Any], str, dict[str, Any], list[str]]:
    config = load_config(config_path)
    keys, combos = grid(config)
    if not 0 <= index < len(combos):
        raise IndexError(f"array index {index} outside 0..{len(combos) - 1}")
    combo = combos[index]
    parameters = dict(zip(keys, combo))
    suffix = (
        "_".join(f"{key.split('.')[-1]}-{safe_piece(value)}" for key, value in parameters.items())
        or "run"
    )
    args: list[str] = []
    fixed = config.get("fixed") or {}
    if not isinstance(fixed, dict):
        raise ValueError("fixed must be a mapping from dotted CLI keys to values")
    for key, value in fixed.items():
        args.extend(tyro_args(key, value))
    for key, value in parameters.items():
        args.extend(tyro_args(key, value))
    return config, suffix, parameters, args


def source_digest() -> str:
    """Calculate one SHA256 value for the program files."""
    digest = hashlib.sha256()
    files = []
    for name in ("torchtitan_dynamic_seq_len", "scripts", "cluster"):
        files.extend(
            path
            for path in (PROJECT_ROOT / name).rglob("*")
            if path.suffix in (".py", ".json", ".slurm")
        )
    for path in sorted(files):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
