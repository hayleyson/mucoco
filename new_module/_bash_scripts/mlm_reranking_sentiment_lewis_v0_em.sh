#!/bin/bash
#SBATCH --job-name=sl_bv0_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/sl_bv0_em_%j.out'
#SBATCH --partition=P1

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export NFS_DIR='/shared/s3/lab07/hyeryung'
# srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 3 \
# --closs_weight 0.167236576878629 \
# --selection_criteria allsat_primary \
# --task sentiment-lewis-compr \
# --num_samples 1 \
# --source_data "${NFS_DIR}/loc_edit/data/Sentiment-and-Style-Transfer/data/yelp/sentiment.test.0" \
# --source_style 'negative' \
# --target_style 'positive' \
# --target_label_ids 1 1 \
# --min_epsilons 0.75 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint/' \
# --output_dir_prefix 'outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/lewis-compr/' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0

srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 5  \
--locate_unit word \
--k_per_location 10 \
--n_iter 3 \
--closs_weight 0.167236576878629 \
--selection_criteria allsat_primary \
--task sentiment-lewis-compr \
--num_samples 1 \
--source_data "${NFS_DIR}/loc_edit/data/Sentiment-and-Style-Transfer/data/yelp/sentiment.test.0" \
--source_style 'negative' \
--target_style 'positive' \
--target_label_ids 1 1 \
--min_epsilons 0.75 \
--wandb_project 'sentiment-decoding' \
--model_paths 'gpt2-large' "${NFS_DIR}/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint" \
--tokenizer_paths 'gpt2-large' "${NFS_DIR}/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint" \
--output_dir_prefix 'outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/lewis-compr/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat
