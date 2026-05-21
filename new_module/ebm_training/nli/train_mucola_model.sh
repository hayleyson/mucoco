#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=nli_energy
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --output='new_module/_slurm_outs/nli_ebm_training_mucola_%j.out'


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun python new_module/ebm_training/nli/train_mucola_model.py 