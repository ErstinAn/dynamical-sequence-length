#!/usr/bin/env bash
# ARC defaults. Another server must supply its own account and GPU settings.
#SBATCH --job-name=seq-1b-handoff
#SBATCH --account=jason_optai-paid
#SBATCH --partition=h200_normal_q
#SBATCH --qos=tc_h200_normal_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00

set -euo pipefail

: "${HANDOFF_CHECKOUT:?Set HANDOFF_CHECKOUT to the frozen checkout.}"
: "${HANDOFF_VENV:?Set HANDOFF_VENV to the new Python environment prefix.}"
: "${HANDOFF_ROOT:?Set HANDOFF_ROOT to persistent project storage.}"
: "${DYNAMIC_SEQ_LEN_ASSETS:?Set DYNAMIC_SEQ_LEN_ASSETS to the standalone assets directory.}"
: "${SLURM_JOB_ID:?Submit this file with sbatch.}"

if [[ $# -eq 0 ]]; then
    printf '%s\n' 'Pass a Python script or Python arguments after the sbatch file.' >&2
    exit 2
fi
if [[ ! -x "$HANDOFF_VENV/bin/python" ]]; then
    printf 'Python is missing: %s/bin/python\n' "$HANDOFF_VENV" >&2
    exit 2
fi
if [[ ! -f "$HANDOFF_CHECKOUT/pyproject.toml" ]]; then
    printf 'The checkout is missing: %s\n' "$HANDOFF_CHECKOUT" >&2
    exit 2
fi

# A venv supplies activate. A standalone Conda prefix needs its bin directory.
if [[ -f "$HANDOFF_VENV/bin/activate" ]]; then
    source "$HANDOFF_VENV/bin/activate"
else
    export PATH="$HANDOFF_VENV/bin:$PATH"
fi
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DYNAMIC_SEQ_LEN_OUTPUT="${DYNAMIC_SEQ_LEN_OUTPUT:-$HANDOFF_ROOT/runs}"
export WANDB_ENTITY=rotation_optimizer
export WANDB_PROJECT=debug
export WANDB_MODE=online
export WANDB_DIR="$HANDOFF_ROOT/wandb"
export HF_HOME="$HANDOFF_ROOT/cache/huggingface"
export TORCH_HOME="$HANDOFF_ROOT/cache/torch"

# ARC deletes TMPDIR at job end. Keep reports and checkpoints in HANDOFF_ROOT.
task_tmp="${TMPDIR:?ARC did not set TMPDIR.}/dynamic-seq-$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="$task_tmp/torchinductor"
export TRITON_CACHE_DIR="$task_tmp/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$DYNAMIC_SEQ_LEN_OUTPUT" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME"
cd "$HANDOFF_CHECKOUT"

"$HANDOFF_VENV/bin/python" - "$DYNAMIC_SEQ_LEN_ASSETS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

dataset = Path(sys.argv[1]) / "data/fineweb_edu_long_30b_shuffled_v1"
progress = json.loads((dataset / "rebuild_progress.json").read_text())
expected_manifest = "4bf6946f7b7f7cbad0799746519baaa9ef2518f974d2239a8df2f184db004806"
if (progress.get("complete") is not True
        or progress.get("manifest_sha256") != expected_manifest
        or hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest() != expected_manifest):
    raise RuntimeError("The token cache is incomplete or its manifest identity differs.")
validation_bytes = (dataset / "validation/manifest.json").read_bytes()
if hashlib.sha256(validation_bytes).hexdigest() != "eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b":
    raise RuntimeError("The fixed validation manifest differs.")
validation = json.loads(validation_bytes)
labels = np.load(dataset / "validation" / validation["arrays"]["labels"]["file"],
                 mmap_mode="r", allow_pickle=False)
if labels.shape != (936, 16384) or np.count_nonzero(labels != -100) != 15335424:
    raise RuntimeError("The fixed validation shape or prediction target count differs.")
marker = dataset / "standalone_ready.json"
if marker.exists():
    receipt = json.loads(marker.read_text())
    if (receipt.get("complete") is not True or receipt.get("labels_shape") != [936, 16384]
            or receipt.get("valid_targets") != 15335424):
        raise RuntimeError("The optional standalone receipt disagrees with the data checks.")
print("PASS: complete token cache, manifest identity, and fixed validation")
PY

date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
module list 2>&1
scontrol show job "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
"$HANDOFF_VENV/bin/python" -c \
    'import sys, torch; print({"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda, "visible_gpus": torch.cuda.device_count()})'
printf 'Command:'
printf ' %q' "$HANDOFF_VENV/bin/python" "$@"
printf '\n'

# One Slurm task starts the driver. The driver starts one process per GPU.
exec srun --ntasks=1 --cpu-bind=none --export=ALL "$HANDOFF_VENV/bin/python" "$@"
