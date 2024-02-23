#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --partition=P1
#SBATCH --gres=gpu:0
#SBATCH --output='new_module/_slurm_outs/beamsearch_fillin_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache

srun python new_module/beamsearch_fillin.py