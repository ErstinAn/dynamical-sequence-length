#!/bin/bash
#SBATCH --job-name=fw-edu-long-12k
#SBATCH --account=jason_optai-paid
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_base
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kangan/titanprecond-lab/dynamic_seq_len/outputs/fineweb_edu_long/supplement-%j.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=true
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export FW_LONG_ROOT=/scratch/kangan/fineweb-edu-long-87f09149/supplement-12k
export FW_LONG_CODE=/home/kangan/titanprecond-lab/dynamic_seq_len
export FW_LONG_PYTHON=/home/kangan/.conda/envs/torchtitan-macro/bin/python
cd "$FW_LONG_ROOT"
"$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/build_fineweb_long.py" plan --root "$FW_LONG_ROOT" --ranks 64 --candidate-tokens 16000000000 --minimum 12288 --source-maximum 16384
srun --kill-on-bad-exit=1 bash -c 'exec "$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/build_fineweb_long.py" extract --root "$FW_LONG_ROOT" --rank "$SLURM_PROCID"'
