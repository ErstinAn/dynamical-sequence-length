#!/bin/bash
#SBATCH --job-name=fw-edu-default-profile
#SBATCH --account=jason_optai-paid
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_base
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/kangan/titanprecond-lab/dynamic_seq_len/outputs/fineweb_edu_long/profile-%j.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export FW_LONG_ROOT=/scratch/kangan/fineweb-edu-long-87f09149
export FW_LONG_CODE=/home/kangan/titanprecond-lab/dynamic_seq_len
export FW_LONG_PYTHON=/home/kangan/.conda/envs/torchtitan-macro/bin/python
cd "$FW_LONG_ROOT"
srun --kill-on-bad-exit=1 bash -c 'exec "$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/profile_fineweb_default.py" worker --root "$FW_LONG_ROOT" --rank "$SLURM_PROCID" --workers 32'
"$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/profile_fineweb_default.py" merge --root "$FW_LONG_ROOT" --nodes 4
