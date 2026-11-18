"""Require finite training and validation metrics before publishing an artifact."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    log = re.sub(r"\x1b\[[0-9;]*m", "", args.log.read_text())
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    sequence_length = manifest["selection"]["minimum_llama3_tokens"]
    observed_lengths = {int(x) for x in re.findall(r"sequence length (\d+)", log)}
    if observed_lengths != {sequence_length}:
        raise ValueError("Training window does not match the selected minimum document length")
    train = [(int(s), float(loss), float(grad)) for s, loss, grad in re.findall(
        r"(?<!validate )step:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)", log)]
    validation = [(int(s), float(loss)) for s, loss in re.findall(
        r"validate step:\s*(\d+)\s+loss:\s*(\S+)", log)]
    if {s for s, _, _ in train} != {1, 2, 3} or not {1, 3}.issubset({s for s, _ in validation}):
        raise ValueError("Missing expected training or validation steps")
    if not all(math.isfinite(x) for _, loss, grad in train for x in [loss, grad]):
        raise ValueError("Non-finite training loss or gradient norm")
    if not all(math.isfinite(loss) for _, loss in validation):
        raise ValueError("Non-finite validation loss")
    receipt = {"passed": True, "model": "upstream Llama 3 1B, vocabulary 128256",
               "sequence_length": sequence_length, "train_steps": train, "validation": validation,
               "manifest_sha256": hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(),
               "log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest()}
    (args.dataset / "training_smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
