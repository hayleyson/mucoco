#!/bin/bash
#SBATCH --job-name=test_ext_eval_subset
#SBATCH --time=00:03:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hyeryung/data/mucoco/laser_edit/_slurm_outs/test_ext_eval_subset_%j.out

set -euo pipefail
source /home/hyeryung/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit
cd /home/hyeryung/data/mucoco
export PYTHONPATH=.
pytest -q tests/evaluation/test_extract_external_evaluator_failure_subsets.py
