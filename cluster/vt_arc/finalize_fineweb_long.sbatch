#!/bin/bash
#SBATCH --job-name=fw-edu-long-finalize
#SBATCH --account=jason_optai-paid
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kangan/titanprecond-lab/dynamic_seq_len/outputs/fineweb_edu_long/finalize-%j.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
FW_LONG_BASE=/scratch/kangan/fineweb-edu-long-87f09149
FW_LONG_ROOT="${FW_LONG_SELECTION_ROOT:-$FW_LONG_BASE/combined-12k}"
FW_LONG_CODE=/home/kangan/titanprecond-lab/dynamic_seq_len
FW_LONG_PYTHON=/home/kangan/.conda/envs/torchtitan-macro/bin/python
FW_LONG_OUTPUT=/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b
if [ -z "${FW_LONG_SELECTION_ROOT:-}" ]; then
    "$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/build_fineweb_long.py" combine --root "$FW_LONG_ROOT" --candidate-root "$FW_LONG_BASE" --candidate-root "$FW_LONG_BASE/supplement-12k"
fi
cd "$FW_LONG_ROOT"
"$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/build_fineweb_long.py" select --root "$FW_LONG_ROOT"
"$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/build_fineweb_long.py" materialize --root "$FW_LONG_ROOT" --output "$FW_LONG_OUTPUT"
"$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/build_fineweb_long.py" verify --root "$FW_LONG_ROOT" --output "$FW_LONG_OUTPUT"
"$FW_LONG_PYTHON" "$FW_LONG_CODE/scripts/summarize_long_artifact.py" --dataset "$FW_LONG_OUTPUT" --report "$FW_LONG_CODE/docs/fineweb-edu-stats/fineweb-edu-long-30b.md"
