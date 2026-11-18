"""Run the measured eight-step smoke and compare uninterrupted and resumed states."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(command, path, env):
    with path.open("w") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Command failed with exit {process.returncode}. Read {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True,
                        help="JSON with microbatches mapping sequence length to measured microbatch.")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--probe", action="store_true",
                        help="Measure microbatches in this allocation before the smoke runs.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "DYNAMIC_SEQ_LEN_ASSETS": str(args.assets.resolve()),
           "PYTHONDONTWRITEBYTECODE": "1"}
    if args.probe:
        run([sys.executable, str(ROOT / "scripts/probe_microbatch.py"),
             "--config", str(args.config.resolve()), "--dp-degree", str(args.nproc),
             "--output", str(args.profile.resolve())], args.output / "microbatch-probe.txt", env)
    profile = json.loads(args.profile.read_text())
    original = yaml.safe_load(args.config.read_text())
    global_batch = original["fixed"]["training.global_batch_size"]
    if (profile.get("status") != "passed" or profile.get("target_dp_degree") != args.nproc
            or profile.get("global_batch_size") != global_batch):
        raise ValueError("Use a passed GPU profile for the requested DP degree and global batch size.")
    microbatches = [int(profile["microbatches"][str(length)]) for length in (2048, 4096, 8192, 16384)]
    if any(micro < 1 or global_batch % (args.nproc * micro) for micro in microbatches):
        raise ValueError(f"Each microbatch times GPU count must divide the global batch of {global_batch}.")
    original["fixed"].update({
        "training.local_batch_size": microbatches[0],
        "microbatch_scheduler.kind": "piecewise_linear",
        "microbatch_scheduler.batch_sizes": [m for m in microbatches for _ in range(2)],
        "microbatch_scheduler.anchor_steps": list(range(1, 9)),
    })
    original["paths"]["dump_root"] = str(args.output / "runs")
    record = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "profile": profile, "microbatches": microbatches, "nproc": args.nproc,
              "token_budget": original["fixed"]["token_budget"], "runs": {}}
    (args.output / "acceptance.json").write_text(json.dumps(record, indent=2) + "\n")
    directories = {}
    for case in ("uninterrupted", "recovered"):
        config = {**original, "name": "smoke-" + case}
        path = args.output / (case + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, str(ROOT / "scripts/train.py"), str(path), "--nproc", str(args.nproc)]
        run([*command, "--dry-run"], args.output / f"{case}-plan.txt", env)
        run([*command, *(["--stop-after-checkpoint", "4"] if case == "recovered" else [])],
            args.output / f"{case}-train.txt", env)
        children = list((args.output / "runs" / config["name"]).iterdir())
        if len(children) != 1:
            raise RuntimeError("The smoke case must have exactly one run directory.")
        directories[case] = children[0]
        if case == "recovered":
            run([sys.executable, str(ROOT / "scripts/train.py"), "--resume-dir", str(children[0])],
                args.output / "recovered-resume.txt", env)
        checkpoints = list((children[0] / "checkpoint").glob("step-*/.metadata"))
        if {p.parent.name for p in checkpoints} != {"step-2", "step-4", "step-6", "step-8"}:
            raise RuntimeError("The smoke did not produce all four complete phase checkpoints.")
        record["runs"][case] = str(children[0])
        (args.output / "acceptance.json").write_text(json.dumps(record, indent=2) + "\n")
    run([sys.executable, str(ROOT / "scripts/compare_checkpoints.py"),
         str(directories["uninterrupted"] / "checkpoint/step-8"),
         str(directories["recovered"] / "checkpoint/step-8"),
         "--output", str(args.output / "checkpoint_comparison.json")],
        args.output / "checkpoint-comparison.txt", env)
    record["checkpoint_comparison"] = json.loads((args.output / "checkpoint_comparison.json").read_text())
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    record["wandb_history_check"] = "pending independent online verification"
    (args.output / "acceptance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
