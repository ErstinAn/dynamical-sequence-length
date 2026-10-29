#!/usr/bin/env bash
set -euo pipefail

SLURM_LAUNCHER_DIR="${SLURM_LAUNCHER_DIR:?Use submit.py to start this script.}"
RUN_CONFIG="${RUN_CONFIG:?Use submit.py to start this script.}"
RUN_MANIFEST="${RUN_MANIFEST:?Use submit.py to start this script.}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON_BIN="${DYNAMIC_SEQ_LEN_PYTHON:-python}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
GPUS_PER_TASK="$("$PYTHON_BIN" -c 'import sys, yaml; print(int(yaml.safe_load(open(sys.argv[1]))["resources"]["gpus_per_node"]))' "$RUN_CONFIG")"
srun --ntasks=1 --cpu-bind=none --gpu-bind="per_task:$GPUS_PER_TASK" --gpus-per-task="$GPUS_PER_TASK" \
  "$PYTHON_BIN" "$SLURM_LAUNCHER_DIR/run_from_yaml.py" "$RUN_CONFIG" \
  --manifest "$RUN_MANIFEST" --index "$TASK_INDEX"
