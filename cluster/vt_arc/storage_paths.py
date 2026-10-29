"""Use the paths from the experiment configuration."""

from pathlib import Path
from typing import Any


def storage_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def runtime_paths(config: dict[str, Any]) -> dict[str, str]:
    return {key: str(storage_path(value)) for key, value in config["paths"].items()}
