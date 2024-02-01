#!/bin/bash
#SBATCH --job-name=data_handling
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --mem=50gb
#SBATCH --output='jigsaw_data_handling_%j.out'

## previously named prepare_training_data_toxicity_binary_dump.sh

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python new_module/em_training/prepare_training_data_toxicity_binary_dump.py