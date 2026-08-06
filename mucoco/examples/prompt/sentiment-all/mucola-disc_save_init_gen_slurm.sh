#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/sentiment_init_gen_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun bash examples/prompt/sentiment-all/mucola-disc_save_init_gen.sh