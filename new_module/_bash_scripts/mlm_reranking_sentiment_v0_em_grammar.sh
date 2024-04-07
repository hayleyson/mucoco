#!/bin/bash
#SBATCH --job-name=s_bv0_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/s_bv0_em_%j.out'
#SBATCH --nodelist=n02

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

# srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 3 \
# --closs_weight 0.167236576878629 \
# --selection_criteria allsat_primary \
# --task sentiment \
# --num_samples 20 \
# --source_data 'new_module/data/sentiment/outputs.txt.init.jsonl' \
# --source_style 'negative' \
# --target_style 'positive' \
# --target_label_ids 1 1 \
# --min_epsilons 0.75 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint/' \
# --output_dir_prefix 'outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm'


srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 5  \
--locate_unit word \
--k_per_location 10 \
--n_iter 3 \
--loss_weights 0.1 0.7 0.2 \
--selection_criteria allsat_primary \
--task sentiment \
--num_samples 20 \
--source_data 'new_module/data/sentiment/dev_set.jsonl' \
--source_style 'negative' \
--target_style 'positive' \
--target_label_ids 1 1 1 \
--min_epsilons 0.9 0.75 \
--wandb_project 'sentiment-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' 'cointegrated/roberta-large-cola-krishna2020'\
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' 'cointegrated/roberta-large-cola-krishna2020'\
--output_dir_prefix 'outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-energy-training/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--losses "gpt2" "classification_no_prefix_logprobloss" "classification_logprobloss" \
--dont_skip_allsat
