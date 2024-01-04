#!/bin/bash
#SBATCH --job-name=formality-ranker
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --mem=50gb
#SBATCH --partition=P1
#SBATCH --output='slurm_output/%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python examples/training_constraint_models/prepare_training_data_sentiment_yelp.py