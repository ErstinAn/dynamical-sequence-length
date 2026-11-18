#!/bin/bash
#SBATCH --job-name=fw-edu-long-supply-watch
#SBATCH --account=jason_optai-paid
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/home/kangan/titanprecond-lab/dynamic_seq_len/outputs/fineweb_edu_long/supply-watch-%j.log
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd /home/kangan/titanprecond-lab/dynamic_seq_len
/home/kangan/.conda/envs/torchtitan-macro/bin/python scripts/watch_fineweb_candidate_supply.py --root /scratch/kangan/fineweb-edu-long-87f09149 --supplement /scratch/kangan/fineweb-edu-long-87f09149/supplement-12k
