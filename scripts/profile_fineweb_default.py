"""Split the pinned default manifest across nodes and merge exact length profiles."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from profile_fineweb_edu_lengths import summarize, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "worker", "merge"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--revision", default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    source = args.root / "source_manifest.json"
    if args.stage == "prepare":
        if args.manifest:
            manifest = json.loads(args.manifest.read_text())
        else:
            from huggingface_hub import HfApi
            info = HfApi(token=False).dataset_info("HuggingFaceFW/fineweb-edu",
                                                 revision=args.revision, files_metadata=True)
            files = [{"path": f.rfilename, "size": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
                     for f in info.siblings if f.rfilename.startswith("data/") and f.rfilename.endswith(".parquet")]
            if not files or any(not f["size"] for f in files):
                raise ValueError("Source manifest is empty or incomplete")
            manifest = {"repo_id": info.id, "revision": info.sha, "config": "default",
                        "files": files, "expected_documents": None}
        if source.exists() and json.loads(source.read_text()) != manifest:
            raise ValueError("Output already belongs to another snapshot")
        write_json(source, manifest)
        # Greedy partition balances compressed source size; exact assignment is frozen.
        groups = [[] for _ in range(args.nodes)]
        loads = [0] * args.nodes
        for entry in sorted(manifest["files"], key=lambda e: (-e["size"], e["path"])):
            rank = min(range(args.nodes), key=lambda i: (loads[i], i))
            groups[rank].append(entry)
            loads[rank] += entry["size"]
        assignments = []
        for rank, files in enumerate(groups):
            folder = args.root / "profile" / f"worker-{rank:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            subset = {**manifest, "files": sorted(files, key=lambda e: e["path"]),
                      "expected_documents": None}
            write_json(folder / "input_manifest.json", subset)
            assignments.append({"rank": rank, "files": len(files), "bytes": loads[rank]})
        write_json(args.root / "profile_assignments.json", assignments)
        print(json.dumps(assignments))
    elif args.stage == "worker":
        folder = args.root / "profile" / f"worker-{args.rank:02d}"
        subprocess.run([sys.executable, str(Path(__file__).with_name("profile_fineweb_edu_lengths.py")),
                        "--source-manifest", str(folder / "input_manifest.json"),
                        "--output", str(folder), "--workers", str(args.workers)], check=True)
    else:
        manifest = json.loads(source.read_text())
        combined = np.zeros(1, dtype=np.int64)
        paths = []
        rows = []
        for rank in range(args.nodes):
            folder = args.root / "profile" / f"worker-{rank:02d}"
            report = json.loads((folder / "report.json").read_text())
            if not report["complete"] or report["revision"] != manifest["revision"]:
                raise ValueError(f"Incomplete or mismatched worker {rank}")
            with np.load(folder / "length_histogram.npz") as data:
                counts = data["counts"]
            if len(counts) > len(combined):
                combined = np.pad(combined, (0, len(counts) - len(combined)))
            combined[:len(counts)] += counts
            paths += report["completed_paths"]
            for entry in json.loads((folder / "input_manifest.json").read_text())["files"]:
                key = entry["path"].removesuffix(".parquet").replace("/", "__")
                rows.append({**entry, "profile_rank": rank,
                             "length_index": str((folder / "length_index" / f"{key}.npy").relative_to(args.root)),
                             "profile_record": str((folder / "shards" / f"{key}.json").relative_to(args.root))})
        if len(paths) != len(set(paths)) or set(paths) != {x["path"] for x in manifest["files"]}:
            raise ValueError("Source coverage is not exact")
        stats = summarize(combined)
        # Include intermediate long thresholds in addition to the earlier 100BT report.
        lengths = np.arange(len(combined), dtype=np.int64)
        extra = []
        for minimum in [8192, 12288, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 65536]:
            extra.append({"minimum_length": minimum, "documents": int(combined[minimum:].sum()),
                          "source_tokens": int(np.dot(combined[minimum:], lengths[minimum:]))})
        report = {"complete": True, "repo_id": manifest["repo_id"], "revision": manifest["revision"],
                  "config": "default", "files": len(paths), "statistics": stats,
                  "additional_thresholds": extra, "length_basis": "source token_count, GPT-2; not Llama 3"}
        np.savez_compressed(args.root / "profile" / "length_histogram.npz", counts=combined)
        write_json(args.root / "profile" / "report.json", report)
        write_json(args.root / "profile" / "indexed_manifest.json", {**manifest, "files": sorted(rows, key=lambda e: e["path"])})
        write_csv(args.root / "profile" / "length_buckets.csv", stats["buckets"])
        write_csv(args.root / "profile" / "length_thresholds.csv", extra)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
