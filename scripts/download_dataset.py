"""Download the pinned dataset and install its tokenizer from this checkout.

This command does not rebuild the training token cache. Run the repository's
rebuild_long_token_cache.py on a CPU compute allocation after this command.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = "long-sequence-length-dataset/fineweb-edu-long-30b"
BUNDLED_TOKENIZER = Path(__file__).resolve().parents[1] / "assets/tokenizer"
DATASET_MANIFEST_SHA256 = "7e284ce4fc90a93e5bdcee160a60051dc29ed17775b8ecb261feafd93f047d80"
TRAINING_MANIFEST_SHA256 = "4bf6946f7b7f7cbad0799746519baaa9ef2518f974d2239a8df2f184db004806"
VALIDATION_MANIFEST_SHA256 = "eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}. Keep the reference hash unchanged.")
    return actual


def standalone_paths(dataset_dir, tokenizer_dir):
    dataset = Path(dataset_dir).expanduser().absolute()
    tokenizer = Path(tokenizer_dir).expanduser().absolute()
    for path in (dataset, dataset / "data", dataset / "tokens", tokenizer):
        if path.is_symlink():
            raise ValueError(f"Use a standalone directory instead of a link: {path}")
    return dataset, tokenizer


def install_bundled_tokenizer(destination, reference):
    """Install the tokenizer identified by the hash-checked dataset manifest."""
    source_record = BUNDLED_TOKENIZER / "tokenizer_source.json"
    if json.loads(source_record.read_text()) != reference:
        raise ValueError("The bundled tokenizer differs from the pinned dataset provenance.")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = dict(reference["files_sha256"])
    for name in ("LICENSE", "Notice", "USE_POLICY.md", "tokenizer_source.json"):
        hashes[name] = sha256(BUNDLED_TOKENIZER / name)
    for name, expected in hashes.items():
        source = BUNDLED_TOKENIZER / name
        require_hash(source, expected)
        target = destination / name
        if target.is_symlink():
            raise ValueError(f"Use a standalone tokenizer file instead of a link: {target}")
        if target.exists():
            require_hash(target, expected)
            continue
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copy(source, temporary)
            require_hash(temporary, expected)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def check_metadata(dataset, tokenizer):
    require_hash(dataset / "dataset_manifest.json", DATASET_MANIFEST_SHA256)
    require_hash(dataset / "manifest.json", TRAINING_MANIFEST_SHA256)
    require_hash(dataset / "validation/manifest.json", VALIDATION_MANIFEST_SHA256)
    shared = json.loads((dataset / "dataset_manifest.json").read_text())
    receipt = json.loads((dataset / "verification.json").read_text())
    if (receipt.get("passed") is not True
            or receipt.get("dataset_manifest_sha256") != DATASET_MANIFEST_SHA256
            or receipt.get("manifest_sha256") != TRAINING_MANIFEST_SHA256):
        raise ValueError("The verification receipt does not match the fixed dataset identity.")
    for entry in shared["document_orders"].values():
        require_hash(dataset / entry["path"], entry["sha256"])
    validation = json.loads((dataset / "validation/manifest.json").read_text())
    if (validation["num_sequences"], validation["sequence_length"], validation["token_budget"]) != (936, 16384, 15335424):
        raise ValueError("The fixed validation dimensions differ from the agreed experiment.")
    for entry in (*validation["arrays"].values(), validation["evaluation_documents"]):
        require_hash(dataset / "validation" / entry["file"], entry["sha256"])
    provenance = json.loads((dataset / "tokenizer_provenance.json").read_text())
    if provenance != shared["tokenizer"]:
        raise ValueError("Tokenizer provenance differs from the pinned dataset manifest.")
    for name, expected in provenance["files_sha256"].items():
        require_hash(tokenizer / name, expected)
    return shared, provenance


def download(dataset_dir, tokenizer_dir, *, revision, cache_dir, metadata_only=False,
             max_workers=4):
    dataset, tokenizer = standalone_paths(dataset_dir, tokenizer_dir)
    cache = Path(cache_dir).expanduser().absolute()
    # Isolate both Hub downloads and Xet chunks from earlier machine caches.
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    previous_path = dataset / "download_receipt.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    resolved = HfApi().dataset_info(REPO_ID, revision=revision).sha
    if previous and previous["revision"] != resolved:
        raise ValueError("The existing download uses another revision. Use a new destination directory.")
    cache_preexisting = cache.exists()
    dataset_preexisting = dataset.exists()
    dataset.mkdir(parents=True, exist_ok=True)
    remote = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", revision=resolved,
                             filename="dataset_manifest.json", cache_dir=cache)
    require_hash(remote, DATASET_MANIFEST_SHA256)
    excluded = ["data/*", "training_code.tar.gz", "training_code.tar.gz.sha256"]
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=resolved,
                      local_dir=dataset, cache_dir=cache, ignore_patterns=excluded,
                      max_workers=max_workers)
    install_bundled_tokenizer(tokenizer, json.loads(Path(remote).read_text())["tokenizer"])
    shared, provenance = check_metadata(dataset, tokenizer)
    text_files = []
    if not metadata_only:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=resolved,
                          local_dir=dataset, cache_dir=cache,
                          allow_patterns=[entry["text"]["path"] for entry in shared["files"]],
                          max_workers=max_workers)
        for entry in shared["files"]:
            text = entry["text"]
            require_hash(dataset / text["path"], text["sha256"])
            text_files.append(text["path"])
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "repo_id": REPO_ID,
        "revision": resolved, "dataset_dir": str(dataset), "tokenizer_dir": str(tokenizer),
        "cache_dir": str(cache), "xet_cache": os.environ["HF_XET_CACHE"],
        "cache_preexisting": cache_preexisting, "dataset_directory_preexisting": dataset_preexisting,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "manifest_sha256": TRAINING_MANIFEST_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "tokenizer_revision": provenance["revision"], "tokenizer_source": "repository:assets/tokenizer",
        "metadata_checks_passed": True,
        "text_files_checked": text_files, "text_download_complete": not metadata_only,
        "token_cache_rebuilt_by_this_command": False,
    }
    temporary = previous_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(previous_path)
    print(json.dumps({key: value for key, value in result.items() if key != "text_files_checked"}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Fixed dataset commit SHA from Hugging Face.")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive.")
    download(args.dataset_dir, args.tokenizer_dir, revision=args.revision,
             cache_dir=args.cache_dir, metadata_only=args.metadata_only,
             max_workers=args.max_workers)


if __name__ == "__main__":
    main()
