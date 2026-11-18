"""Download matching Hub text and rebuild a standalone fixed-order dataset."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def restore(root, tokenizer, *, revision=None):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    root, tokenizer = Path(root).resolve(), Path(tokenizer).resolve()
    shared = json.loads((root / "dataset_manifest.json").read_text())
    receipt = json.loads((root / "verification.json").read_text())
    actual = hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest()
    if not receipt["passed"] or receipt["dataset_manifest_sha256"] != actual:
        raise ValueError("Dataset metadata does not match its verification receipt")
    if any((root / name).is_symlink() for name in ("data", "tokens")):
        raise ValueError("Restore into a standalone package, not the local linked copy")
    source = json.loads((root / "source_upload_receipt.json").read_text())
    provenance = json.loads((root / "tokenizer_provenance.json").read_text())
    # The source receipt records provenance. Its old commit can be removed from
    # Hub history. Resolve the requested current revision once for this restore.
    resolved = HfApi().dataset_info(source["repo_id"], revision=revision).sha
    remote = hf_hub_download(repo_id=source["repo_id"], repo_type="dataset", revision=resolved,
                             filename="dataset_manifest.json")
    if hashlib.sha256(Path(remote).read_bytes()).hexdigest() != actual:
        raise ValueError("Hub dataset metadata differs from this package; download the matching package first")
    snapshot_download(repo_id=source["repo_id"], repo_type="dataset", revision=resolved,
                      allow_patterns=[f["text"]["path"] for f in shared["files"]], local_dir=root)
    snapshot_download(repo_id=provenance["repo_id"], revision=provenance["revision"],
                      allow_patterns=list(provenance["files_sha256"]), local_dir=tokenizer)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.rebuild_long_token_cache import rebuild
    expected = receipt["manifest_sha256"]
    rebuild(root, tokenizer)
    if hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != expected:
        raise ValueError("Rebuilt dataset manifest differs from the shared identity")
    print(json.dumps({"restored": str(root), "manifest_sha256": expected, "hub_revision": resolved}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--revision", help="Hub dataset revision; defaults to current main")
    args = parser.parse_args()
    restore(args.dataset_dir, args.tokenizer_dir, revision=args.revision)
