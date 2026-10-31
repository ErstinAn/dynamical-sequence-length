"""Run GPU comparisons in an allocation with one compute node."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def run(name, *extra):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc}",
            "-m",
            "tests.dynamic_seq_gpu",
            "--dump",
            str(args.output / name),
            *extra,
        ]
        with (args.output / f"{name}.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        return torch.load(args.output / name / "result.pt", weights_only=False)

    def identical(left, right):
        assert left["model"].keys() == right["model"].keys()
        for name in left["model"]:
            torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
        assert left["trainer"]["ntokens_seen"] == right["trainer"]["ntokens_seen"]
        assert left["trainer"]["step"] == right["trainer"]["step"]

    native = run("native", "--native")
    constant = run("constant", "--constant")
    identical(native, constant)
    assert constant["trainer"]["n_valid_tokens_seen"] == 4096
    full = run("triangular")
    assert full["trainer"]["n_valid_tokens_seen"] == 2304
    for step in (2, 4):
        name = f"resumed-{step}"
        shutil.copytree(
            args.output / "triangular/checkpoint" / f"step-{step}",
            args.output / name / "checkpoint" / f"step-{step}",
        )
        resumed = run(name, "--load-step", str(step))
        identical(full, resumed)
        assert full["trainer"] == resumed["trainer"]
    compiled = run("compiled", "--compile")
    assert compiled["trainer"] == full["trainer"]
    # Compilation can change the order of floating-point operations.
    # Uncompiled runs and their resumed runs must have the same results.
    # TorchTitan and constant dynamic runs must also have the same results.
    for name, parameter in compiled["model"].items():
        assert torch.isfinite(parameter).all(), name
    result = dict(
        nproc=args.nproc,
        native_constant_exact=True,
        resume_steps=[2, 4],
        resume_exact=True,
        triangular_valid_tokens=2304,
        compile_completed=True,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
