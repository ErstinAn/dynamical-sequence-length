#!/usr/bin/env bash
#SBATCH --job-name=seq-grad-repair-check
#SBATCH --account=jason_optai-paid
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_CACHE_DIR="${SEQ_PIPELINE_ROOT}/inductor_cache"
export TRITON_CACHE_DIR="${SEQ_PIPELINE_ROOT}/triton_cache"
export TOKENIZERS_PARALLELISM=false
cd "${SEQ_PIPELINE_ROOT}/source"
srun --ntasks=1 --cpu-bind=none --gpu-bind=per_task:1 --gpus-per-task=1 \
 /home/kangan/.conda/envs/torchtitan-macro/bin/python -m torch.distributed.run \
 --standalone --nproc_per_node=1 -m tests.seq_grad_microbatch_gpu \
 --output "${SEQ_PIPELINE_ROOT}/single_gpu_repair_check"
