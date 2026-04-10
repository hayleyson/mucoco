#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_infer
#SBATCH --output='new_module/_slurm_outs/prm800k_p2_train_infer_%j.out'
#SBATCH --nodelist=n02


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export HF_DATASETS_CACHE=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

srun python new_module/_notebooks/nb_20251214_ebm_prm800k.py