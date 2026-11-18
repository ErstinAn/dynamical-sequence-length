"""Resolve explicitly archived local assets without substituting dataset versions."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_archived_asset(path):
    requested = Path(path).expanduser()
    if requested.exists() or requested.absolute().parent != ROOT / "assets" / "data":
        return requested
    inventory = ROOT / "archive" / "data" / "manifest.json"
    if not inventory.is_file():
        return requested
    entry = json.loads(inventory.read_text()).get("entries", {}).get(requested.name)
    if entry is None:
        return requested
    destination = (inventory.parent / entry["directory"]).resolve()
    if not destination.is_relative_to(inventory.parent.resolve()):
        raise ValueError("Archived data path escapes its archive")
    if not destination.is_dir():
        raise FileNotFoundError(f"Archived dataset is missing: {destination}")
    return destination
