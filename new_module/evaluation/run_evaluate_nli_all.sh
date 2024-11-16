#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-20:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=llm_edit_eval
#SBATCH --output='/data3/saeheeeom/set_consistency/mucoco/new_module/_slurm_outs/edited_nli_eval_%j.out'

source /data3/saeheeeom/.bashrc
source /data3/saeheeeom/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data3/saeheeeom/.cache
export HF_DATASETS_CACHE=/data3/saeheeeom/.cache
export TRANSFORMERS_CACHE=/data3/saeheeeom/.cache
export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID

srun python /data3/saeheeeom/set_consistency/mucoco/new_module/evaluation/evaluate_nli_all.py