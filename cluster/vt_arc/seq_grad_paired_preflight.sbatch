#!/usr/bin/env bash
#SBATCH --job-name=seq-paired-preflight
#SBATCH --account=jason_optai-paid
#SBATCH --partition=h200_normal_q
#SBATCH --qos=tc_h200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SEQ_PIPELINE_ROOT}/preflight_inductor_cache"
export TRITON_CACHE_DIR="${SEQ_PIPELINE_ROOT}/preflight_triton_cache"
export TOKENIZERS_PARALLELISM=false
cd "${SEQ_PIPELINE_ROOT}/source"
srun --ntasks=1 --cpu-bind=none --gpu-bind=per_task:1 --gpus-per-task=1 \
 /home/kangan/.conda/envs/torchtitan-macro/bin/python -m tests.seq_grad_single_document_gpu \
 --output "${SEQ_PIPELINE_ROOT}/preflight_single_document_attention.json"
srun --ntasks=1 --cpu-bind=none --gpu-bind=per_task:1 --gpus-per-task=1 \
 /home/kangan/.conda/envs/torchtitan-macro/bin/python -m torch.distributed.run \
 --standalone --nproc_per_node=1 -m tests.seq_grad_microbatch_gpu \
 --sampling paired_window --output "${SEQ_PIPELINE_ROOT}/preflight_state_isolation"
