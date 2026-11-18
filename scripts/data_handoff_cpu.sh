#!/usr/bin/env bash
# Run inside a Slurm allocation with 32 CPUs and 128 GiB of memory.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDOFF_DIR="${HANDOFF_DIR:-/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118}"
DATASET_DIR="$HANDOFF_DIR/assets/data/fineweb_edu_long_30b_shuffled_v1"
TOKENIZER_DIR="$HANDOFF_DIR/assets/hf/llama3-tokenizer"
PYTHON_BIN="$HANDOFF_DIR/environment/conda/bin/python"
DATASET_REVISION=b6ea2fde77bab36dff963aba5e6f482477959933
SHARD_WORKERS=4
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ "${SLURM_CPUS_PER_TASK:-0}" -lt 32 ]]; then
  printf '%s\n' 'This wrapper requires a compute allocation with at least 32 CPUs.' >&2
  exit 2
fi
cd "$REPO_DIR"
mkdir -p "$HANDOFF_DIR/logs"
exec > >(tee -a "$HANDOFF_DIR/logs/data_handoff_${SLURM_JOB_ID:-manual}.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
"$PYTHON_BIN" -c 'import sys, numpy, pyarrow, tokenizers, huggingface_hub; print({"python": sys.executable, "version": sys.version, "numpy": numpy.__version__, "pyarrow": pyarrow.__version__, "tokenizers": tokenizers.__version__, "huggingface_hub": huggingface_hub.__version__})'
sha256sum scripts/download_dataset.py scripts/rebuild_long_token_cache.py scripts/data_handoff_cpu.sh
"$PYTHON_BIN" -m unittest tests.test_download_dataset tests.test_rebuild_long_token_cache -v
"$PYTHON_BIN" scripts/download_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --cache-dir "$HANDOFF_DIR/download-cache" \
  --revision "$DATASET_REVISION" \
  --max-workers 4

# The benchmark uses the complete first shard. It cannot publish a complete cache.
"$PYTHON_BIN" scripts/rebuild_long_token_cache.py \
  --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
  --workers 1 --limit-shards 1
cp "$DATASET_DIR/rebuild_progress.json" "$HANDOFF_DIR/first_shard_benchmark.json"
"$PYTHON_BIN" - "$DATASET_DIR" "$SHARD_WORKERS" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
workers = int(sys.argv[2])
benchmark = json.loads((root / "rebuild_progress.json").read_text())
shared = json.loads((root / "dataset_manifest.json").read_text())
sample = benchmark["shards"][0]
rate = sample["tokens"] / sample["elapsed_seconds"]
remaining = sum(entry["tokens"] for entry in shared["files"]) - sample["tokens"]
print(json.dumps({"benchmark_tokens_per_second": rate,
                  "remaining_tokens": remaining,
                  "ideal_parallel_remaining_hours": remaining / rate / workers / 3600,
                  "estimate_basis": "One complete shard. Ideal worker scaling. Actual runtime can be longer."}), flush=True)
PY

# Each resumed shard must pass source, token, and index hash checks.
"$PYTHON_BIN" scripts/rebuild_long_token_cache.py \
  --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
  --workers "$SHARD_WORKERS" --resume
"$PYTHON_BIN" - "$DATASET_DIR" "$TOKENIZER_DIR" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
from scripts.download_dataset import check_metadata, sha256, TRAINING_MANIFEST_SHA256
root, tokenizer = map(Path, sys.argv[1:])
shared, provenance = check_metadata(root, tokenizer)
progress = json.loads((root / "rebuild_progress.json").read_text())
assert progress["complete"] is True
assert progress["manifest_sha256"] == TRAINING_MANIFEST_SHA256
for split, entry in shared["document_orders"].items():
    order = np.load(root / entry["path"], mmap_mode="r")
    assert np.array_equal(np.sort(order), np.arange(len(order)))
validation = json.loads((root / "validation/manifest.json").read_text())
labels = np.load(root / "validation" / validation["arrays"]["labels"]["file"], mmap_mode="r")
assert labels.shape == (936, 16384)
assert np.count_nonzero(labels != -100) == 15335424
result = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
          "complete": True, "shards": len(shared["files"]),
          "manifest_sha256": sha256(root / "manifest.json"),
          "labels_shape": list(labels.shape), "valid_targets": int(np.count_nonzero(labels != -100)),
          "dataset_dir": str(root), "tokenizer_revision": provenance["revision"]}
(root / "standalone_ready.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
PY
date -u +'%Y-%m-%dT%H:%M:%SZ'
