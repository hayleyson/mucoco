#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P2
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
# --outputs_file_0='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-9fjc2pfg-negative-to-positive/outputs_epsilon0.75.txt' \
# --outputs_file_1='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-qdgh9mv2-positive-to-negative/outputs_epsilon0.75.txt' \
# --results_file='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-epsilon0.75-results.txt'

# srun python new_module/compr/evaluate_lewis_metrics.py \
# --outputs_file_0='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-9fjc2pfg-negative-to-positive/outputs_epsilon0.75.txt' \
# --results_file='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-epsilon0.75-results-negative-to-positive.txt'

# srun python new_module/compr/evaluate_lewis_metrics.py \
# --outputs_file_1='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-qdgh9mv2-positive-to-negative/outputs_epsilon0.75.txt' \
# --results_file='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-epsilon0.75-results-positive-to-negative.txt'


srun python new_module/compr/evaluate_lewis_metrics.py \
--outputs_file_0='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-866hmm6j-negative-to-positive-grad_norm-866hmm6j/outputs_epsilon0.75.txt' \
--outputs_file_1='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-ij4hre36-positive-to-negative-grad_norm-ij4hre36/outputs_epsilon0.75.txt' \
--results_file='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-grad_norm-epsilon0.75-results.txt'

srun python new_module/compr/evaluate_lewis_metrics.py \
--outputs_file_0='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-866hmm6j-negative-to-positive-grad_norm-866hmm6j/outputs_epsilon0.75.txt' \
--results_file='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-866hmm6j-negative-to-positive-grad_norm-866hmm6j/results_epsilon0.75.txt' \
--wandb_run_path='hayleyson/sentiment-decoding/866hmm6j' \
--update_wandb

srun python new_module/compr/evaluate_lewis_metrics.py \
--outputs_file_1='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-ij4hre36-positive-to-negative-grad_norm-ij4hre36/outputs_epsilon0.75.txt' \
--results_file='outputs/sentiment/mlm-reranking/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-ij4hre36-positive-to-negative-grad_norm-ij4hre36/results_epsilon0.75.txt' \
--wandb_run_path='hayleyson/sentiment-decoding/ij4hre36' \
--update_wandb