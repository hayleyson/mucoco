#!/bin/bash
#SBATCH --job-name=formality_transfer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=50gb
#SBATCH --partition=P1
#SBATCH --output='new_module/examples/formality_transfer_240105_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python new_module/examples/formality_transfer.py