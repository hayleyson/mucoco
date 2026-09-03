#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --job-name=subset_intersect
#SBATCH --output='/home/hyeryung/data/mucoco/laser_edit/ebm_training/subset_intersection_%j.out'

set -euo pipefail
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

ROOT="/home/hyeryung/data/mucoco"
cd "$ROOT"
export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python laser_edit/ebm_training/failure_union_utils.py
