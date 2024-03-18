#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --gres=gpu:0
#SBATCH --output='new_module/em4mlm_eval_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit


export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache

myArray=("f108tpx2" "4928q9xt")
wandb_project="toxicity-decoding"

## submit multiple sbatch simultaneously
for id in ${myArray[@]}; do         
    sbatch new_module/evaluate_wandb_post_run_persp.sh ${id} ${wandb_project}
done