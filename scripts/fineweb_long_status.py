"""Show persisted progress and live Slurm outcomes for the long-data pipeline."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    code = Path(__file__).resolve().parents[1]
    record = json.loads((code / "outputs/fineweb_edu_long/jobs.json").read_text())
    root = Path(record["root"])
    state = {"updated_at_utc": datetime.now(timezone.utc).isoformat()}
    profiles = [json.loads(p.read_text()) for p in root.glob("profile/worker-*/progress.json")]
    state["profile"] = {"complete_files": sum(p["completed_files"] for p in profiles),
                        "expected_files": sum(p["expected_files"] for p in profiles),
                        "failures": sum(len(p["failed_files"]) for p in profiles),
                        "documents": sum((p.get("statistics") or {}).get("documents", 0) for p in profiles)}
    ranks = [json.loads(p.read_text()) for p in root.glob("extract_status/rank-*.json")]
    state["extract"] = {"reporting_ranks": len(ranks), "complete_ranks": sum(p["complete"] for p in ranks),
                        "files": sum(len(p["files"]) for p in ranks),
                        "candidate_tokens": sum(p["llama3_tokens"] for p in ranks)}
    if record.get("supplement_root"):
        supplemental = [json.loads(p.read_text()) for p in Path(record["supplement_root"]).glob("extract_status/rank-*.json")]
        state["supplement_extract"] = {
            "reporting_ranks": len(supplemental), "complete_ranks": sum(p["complete"] for p in supplemental),
            "files": sum(len(p["files"]) for p in supplemental),
            "candidate_tokens": sum(p["llama3_tokens"] for p in supplemental)}
    artifacts = ["selection.json", "manifest.json", "verification.json", "training_smoke.json", "upload_receipt.json"]
    state["artifacts"] = {name: (Path(record["artifact"]) / name).exists() for name in artifacts}
    selected = Path(record.get("selection_root", str(root))) / "selection.json"
    if selected.exists():
        selection = json.loads(selected.read_text())
        state["selection"] = {"minimum_llama3_tokens": selection["minimum_llama3_tokens"],
                              "splits": selection["splits"]}
    materializing = Path(record["artifact"]) / "materialize_progress.json"
    if materializing.exists():
        progress = json.loads(materializing.read_text())
        state["materialization"] = {"completed_shards": len(progress["files"]),
                                    "tokens": sum(f["tokens"] for f in progress["files"])}
    verifying = Path(record["artifact"]) / "verify_progress.json"
    if verifying.exists():
        state["verification"] = json.loads(verifying.read_text())["totals"]
    supply = root / "supply-watch/status.json"
    if supply.exists():
        progress = json.loads(supply.read_text())
        state["supply"] = {"files_checked": progress["files"],
            "provisional_train_tokens": {str(t["minimum"]): t["splits"].get("train", {}).get("tokens", 0)
                                          for t in progress["provisional_thresholds"] if t["minimum"] in {16384, 12288}},
            "frozen_selection_ready": (root / "supply-watch/ready.json").exists()}
    ids = list(record["stages"].values()) + list(record["checks"].values())
    command = ["sacct", "-X", "-j", ",".join(map(str, ids)), "-P",
               "--format=JobIDRaw,State,ExitCode,Submit,Start,End,Timelimit,Elapsed,AllocCPUS"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = result.stdout.strip().splitlines()
    keys = rows[0].split("|")
    state["jobs"] = [dict(zip(keys, row.split("|"))) for row in rows[1:]]
    target = code / "outputs/fineweb_edu_long/status.json"
    target.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({k: v for k, v in state.items() if k != "jobs"}, indent=2))
    for name, job in record["stages"].items():
        found = next((r for r in state["jobs"] if r["JobIDRaw"] == str(job)), {})
        label = found.get("State", "UNKNOWN")
        if job in record.get("stopped_after_sufficient_selection", []) and label.startswith("CANCELLED"):
            label += " (budget met)"
        print(name, job, label, found.get("Elapsed", ""))


if __name__ == "__main__":
    main()
