#!/usr/bin/env bash
# Submit after the preceding preparation job ends. Do not run both jobs together.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDOFF_DIR="${HANDOFF_DIR:-/projects/jason_optai/kangan/dynamic_seq_len_handoff_20261118}"
DATASET_DIR="$HANDOFF_DIR/assets/data/fineweb_edu_long_30b_shuffled_v1"
TOKENIZER_DIR="$HANDOFF_DIR/assets/hf/llama3-tokenizer"
PYTHON_BIN="$HANDOFF_DIR/environment/conda/bin/python"
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
exec > >(tee -a "$HANDOFF_DIR/logs/data_resume_${SLURM_JOB_ID:-manual}.log") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
sha256sum scripts/download_dataset.py scripts/rebuild_long_token_cache.py scripts/resume_data_handoff_cpu.sh
if [[ -f "$DATASET_DIR/rebuild_progress.json" ]]; then
  cp "$DATASET_DIR/rebuild_progress.json" "$HANDOFF_DIR/rebuild_before_resume_${SLURM_JOB_ID:-manual}.json"
fi
# A prior readiness marker cannot describe this new check until the check passes.
rm -f "$DATASET_DIR/standalone_ready.json"
"$PYTHON_BIN" - "$DATASET_DIR" "$TOKENIZER_DIR" <<'PY'
import json
from pathlib import Path
import sys
from scripts.download_dataset import check_metadata
root, tokenizer = map(Path, sys.argv[1:])
receipt = json.loads((root / "download_receipt.json").read_text())
assert receipt["revision"] == "b6ea2fde77bab36dff963aba5e6f482477959933"
assert receipt["text_download_complete"] is True
assert len(receipt["text_files_checked"]) == 236
check_metadata(root, tokenizer)
print("Complete download receipt and metadata checks passed.", flush=True)
PY

# If every shard is complete, this checks hashes and does not encode documents.
"$PYTHON_BIN" scripts/rebuild_long_token_cache.py \
  --dataset-dir "$DATASET_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
  --workers 4 --resume
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
for entry in shared["document_orders"].values():
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
          "dataset_dir": str(root), "tokenizer_revision": provenance["revision"],
          "checked_reused_shards": sum(s["reused_checked_cache"] for s in progress["shards"]),
          "encoded_shards_in_this_continuation": sum(not s["reused_checked_cache"] for s in progress["shards"])}
temporary = root / "standalone_ready.json.tmp"
temporary.write_text(json.dumps(result, indent=2) + "\n")
temporary.replace(root / "standalone_ready.json")
print(json.dumps(result, indent=2), flush=True)
PY
date -u +'%Y-%m-%dT%H:%M:%SZ'
