#!/usr/bin/env python3
"""Read-only overnight Slurm/log watcher. The active agent handles any repairs."""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=25)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"exit_code": None, "output": "", "error": f"Observation failed: {exc}"}
    return {"exit_code": result.returncode, "output": result.stdout.strip(),
            "error": result.stderr.strip()}


def tail(path):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 20000))
        lines = stream.read().decode(errors="replace").splitlines()
    interesting = [line for line in lines if re.search(
        r"Sequence (probe|event)|\bstep[:= ]+\d|PASS:|Traceback|Error:|FAILED|Training completed", line)]
    return {"path": str(path), "bytes": path.stat().st_size, "mtime": path.stat().st_mtime,
            "latest_progress": interesting[-4:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings", type=Path)
    args = parser.parse_args()
    previous = None
    while True:
        cfg = json.loads(args.settings.read_text())
        output = args.settings.parent
        now = datetime.now().astimezone()
        jobs = ",".join(map(str, cfg["jobs"]))
        result = {"observed_at": now.isoformat(), "deadline": cfg["deadline"],
            "queue": command(["squeue", "-M", cfg["cluster"], "-j", jobs, "--start", "-h",
                              "-o", "%i|%T|%S|%M|%l|%R"]),
            "accounting": command(["sacct", "-M", cfg["cluster"], "-j", jobs, "-n", "-P",
                                   "--format=JobID,State,ExitCode,Start,End,Elapsed,Timelimit"]),
            "logs": [item for path in cfg["logs"] if (item := tail(Path(path)))],
            "pipeline": None}
        status = Path(cfg["pipeline_status"])
        if status.exists():
            result["pipeline"] = json.loads(status.read_text())
        tmp = output / "overnight_status.tmp"
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(output / "overnight_status.json")
        with (output / "overnight_history.jsonl").open("a") as history:
            history.write(json.dumps(result) + "\n")
        signature = json.dumps([result["queue"]["output"], result["accounting"]["output"],
                                result["pipeline"], [log["latest_progress"] for log in result["logs"]]])
        if signature != previous:
            print(json.dumps(result), flush=True)
            previous = signature
        if now >= datetime.fromisoformat(cfg["deadline"]):
            print("WATCH_DEADLINE_REACHED", flush=True)
            return
        failed = re.search(r"\|(FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL)\|", result["accounting"]["output"])
        if failed or (result["pipeline"] or {}).get("state") in {"failed", "withheld"}:
            print("WATCH_ATTENTION_REQUIRED", flush=True)
            return
        if (result["pipeline"] or {}).get("state") == "completed" and all(
            re.search(rf"^{job}\|COMPLETED\|", result["accounting"]["output"], re.MULTILINE)
            for job in cfg["jobs"]
        ):
            print("WATCH_COMPLETED", flush=True)
            return
        time.sleep(50)


if __name__ == "__main__":
    main()
