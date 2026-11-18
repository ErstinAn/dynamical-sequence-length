"""Copy verified token files to node storage before an X monitor run."""

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def stage(source, destination, report_path):
    started = time.perf_counter()
    encoded = (source / "manifest.json").read_bytes()
    manifest = json.loads(encoded)
    identity = hashlib.sha256(encoded).hexdigest()
    verification = json.loads((source / "verification.json").read_text())
    if not manifest["complete"] or not verification["passed"]:
        raise ValueError("Source dataset is incomplete or unverified")
    if verification["manifest_sha256"] != identity:
        raise ValueError("Source verification does not match the manifest")
    destination.mkdir(parents=True, exist_ok=False)
    entries = [item[key] for item in manifest["files"] for key in ("token_ids", "index")]
    entries += list(manifest.get("document_orders", {}).values())
    validation = json.loads((source / "validation/manifest.json").read_text())
    entries += [
        {"path": "validation/" + item["file"], "sha256": item["sha256"]}
        for item in [*validation["arrays"].values(), validation["evaluation_documents"]]
    ]
    needed = sum((source / entry["path"]).stat().st_size for entry in entries)
    if shutil.disk_usage(destination).free < needed * 1.1:
        raise ValueError("Local disk has insufficient free space")

    def copy(entry):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dataset file path must remain inside the dataset")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        shutil.copyfile(source / relative, temporary)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Local file checksum mismatch: {relative}")
        temporary.rename(target)
        return target.stat().st_size

    copied = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(copy, entry) for entry in entries]
        for count, future in enumerate(as_completed(futures), 1):
            copied += future.result()
            if count % 32 == 0 or count == len(entries):
                print(f"Local data: {count}/{len(entries)} files; {copied / 1e9:.1f} GB", flush=True)
    for relative in ("manifest.json", "verification.json", "validation/manifest.json"):
        shutil.copyfile(source / relative, destination / relative)
    report = dict(passed=True, source=str(source), destination=str(destination),
                  manifest_sha256=identity, verified_files=len(entries), bytes=copied,
                  seconds=time.perf_counter() - started)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    stage(args.source, args.destination, args.report)
