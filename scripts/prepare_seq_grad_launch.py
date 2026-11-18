#!/usr/bin/env python3
"""Freeze code/configs and prepare collision-safe smoke/full run manifests."""

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cluster/vt_arc"))
from submit import atomic_write_submission, make_manifest
from torchtitan_dynamic_seq_len.run_config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke-config", type=Path,
                        default=Path("legacy/configs/seq_grad/variance_monitor_paired_document_train16384_smoke.yaml"))
    parser.add_argument("--full-config", type=Path,
                        default=Path("legacy/configs/seq_grad/variance_monitor_paired_document_train16384.yaml"))
    args = parser.parse_args()
    configurations = {}
    for phase, path in (("smoke", args.smoke_config), ("full", args.full_config)):
        original = (ROOT / path).resolve()
        if not original.is_file() or not original.is_relative_to(ROOT):
            parser.error(f"{phase} config must be an existing file in {ROOT}")
        configurations[phase] = original.relative_to(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in ("torchtitan_dynamic_seq_len", "scripts", "cluster", "tests", "configs"):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "assets").symlink_to(ROOT / "assets", target_is_directory=True)
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, source / name)
    plan = {}
    for phase, relative in configurations.items():
        config_path = source / relative
        # Expand roots before recording the manifest so the snapshot can run alone.
        config = load_config(config_path)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = make_manifest(config_path, "new", [uuid.uuid4().hex])
        manifest["slurm_time"] = config["resources"]["time"]
        saved, snapshot = atomic_write_submission(output / "manifests", manifest, config)
        plan[phase] = {"name": phase, "config": str(snapshot), "manifest": str(saved), **manifest["runs"][0]}
    plan["source"] = str(source)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
