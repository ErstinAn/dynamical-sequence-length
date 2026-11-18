#!/usr/bin/env bash
#SBATCH --job-name=variance-monitor-100m
#SBATCH --account=jason_optai-paid
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_CACHE_DIR="${X_PIPELINE_ROOT}/inductor_cache"
cd "${X_SOURCE_ROOT}"
srun --ntasks=1 --cpu-bind=none --gpu-bind=per_task:4 --gpus-per-task=4 \
 /home/kangan/.conda/envs/torchtitan-macro/bin/python scripts/run_x_monitor_pipeline.py "${X_PIPELINE_ROOT}/plan.json"
