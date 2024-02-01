#!/bin/bash
#SBATCH --job-name=mucola-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --partition=P2
#SBATCH --output='slurm_output/%j.out'


start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
# conda activate hson-mucola
conda activate loc-edit


srun bash examples/training_constraint_models/train_sentiment_classifiers.sh sst2

end=$(date +%s)