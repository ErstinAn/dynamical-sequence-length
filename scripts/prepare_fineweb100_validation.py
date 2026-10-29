#!/usr/bin/env python3
"""Freeze the held-out FineWeb 100BT shard into a portable validation artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torchtitan_dynamic_seq_len.data.fineweb100 import DEFAULT_TOKEN_BUDGET, prepare_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "assets/hf/llama-tokenizer")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()
    metadata = prepare_validation(
        args.source_parquet, args.output, args.tokenizer, args.token_budget, args.seq_len
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
