#!/bin/bash
#SBATCH --job-name=fw-edu-long-1b-smoke
#SBATCH --account=jason_optai-paid
#SBATCH --partition=h200_normal_q
#SBATCH --qos=tc_h200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/home/kangan/titanprecond-lab/dynamic_seq_len/outputs/fineweb_edu_long/train-%j.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export OUTPUT_ROOT=/scratch/kangan/fineweb-edu-long-87f09149
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
cd /home/kangan/titanprecond-lab/dynamic_seq_len
FW_LONG_TRAIN_LOG="$OUTPUT_ROOT/training-$SLURM_JOB_ID.log"
/home/kangan/.conda/envs/torchtitan-macro/bin/python scripts/train.py /projects/jason_optai/kangan/datasets/fineweb-edu-long-30b/training_smoke.yaml --nproc 1 2>&1 | tee "$FW_LONG_TRAIN_LOG"
/home/kangan/.conda/envs/torchtitan-macro/bin/python scripts/check_long_training_log.py --log "$FW_LONG_TRAIN_LOG" --dataset /projects/jason_optai/kangan/datasets/fineweb-edu-long-30b
