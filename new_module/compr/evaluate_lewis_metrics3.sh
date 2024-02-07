#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/eval_lewis_metrics_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache

# srun python new_module/compr/evaluate_lewis_metrics.py \
# --outputs_file_0='lewis/model_outputs/yelp/pos.out' \
# --outputs_file_1='lewis/model_outputs/yelp/neg.out' \
# --results_file='lewis/model_outputs/yelp/results.txt'

# srun python new_module/compr/evaluate_lewis_metrics.py \
# --outputs_file_0='lewis/model_outputs/yelp/pos.out' \
# --outputs_file_1='lewis/model_outputs/yelp/neg.out' \
# --results_file='lewis/model_outputs/yelp/results-negative-to-positive.txt'

srun python new_module/compr/evaluate_lewis_metrics.py \
--outputs_file_0='lewis/model_outputs/yelp/pos.out' \
--outputs_file_1='lewis/model_outputs/yelp/neg.out' \
--results_file='lewis/model_outputs/yelp/results-positive-to-negative.txt'