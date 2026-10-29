#!/usr/bin/env python3
"""Download the tokenizer files and check their SHA256 values."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/hf/llama-tokenizer")
    args = parser.parse_args()
    reference = ROOT / "torchtitan_dynamic_seq_len/data/manifests/tokenizer_source.json"
    source = json.loads(reference.read_text())
    files = {}
    for name, digest in source["files_sha256"].items():
        path = Path(
            hf_hub_download(
                repo_id=source["repo_id"],
                revision=source["revision"],
                filename=name,
            )
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"The SHA256 value does not match: {name}")
        files[name] = path
    args.output.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        shutil.copyfile(path, args.output / name)
    shutil.copyfile(reference, args.output / "tokenizer_source.json")
    print(args.output)


if __name__ == "__main__":
    main()
