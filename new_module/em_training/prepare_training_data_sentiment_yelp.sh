#!/bin/bash
#SBATCH --job-name=data_handling
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --mem=50gb
#SBATCH --output='yelp_data_handling_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python new_module/em_training/prepare_training_data_sentiment_yelp.py