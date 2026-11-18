#!/bin/bash
#SBATCH --job-name=fw-edu-long-upload
#SBATCH --account=jason_optai-paid
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kangan/titanprecond-lab/dynamic_seq_len/outputs/fineweb_edu_long/upload-%j.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/kangan/titanprecond-lab/dynamic_seq_len
/home/kangan/.conda/envs/torchtitan-macro/bin/python scripts/build_fineweb_long.py upload --root /scratch/kangan/fineweb-edu-long-87f09149 --output /projects/jason_optai/kangan/datasets/fineweb-edu-long-30b --repo-id long-sequence-length-dataset/fineweb-edu-long-30b
/home/kangan/.conda/envs/torchtitan-macro/bin/python scripts/summarize_long_artifact.py --dataset /projects/jason_optai/kangan/datasets/fineweb-edu-long-30b --report docs/fineweb-edu-stats/fineweb-edu-long-30b.md
