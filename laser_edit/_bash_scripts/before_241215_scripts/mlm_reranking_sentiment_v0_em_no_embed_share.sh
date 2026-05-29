#!/bin/bash
#SBATCH --job-name=s_bv0_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output='laser_edit/_slurm_outs/s_bv0_em_%j.out'
#SBATCH --nodelist=n02

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# !!!!! ATTENTION
# '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint' \
# above model is trained using margin ranking loss, which I didn't fully explore, so not sure if stable.
# sadly, there's no regression model trained without embedding sharing for sentiment task.
# so for sentiment, we'll just use embedding sharing version of regression model for experiments. :/

# srun python laser_edit/mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 3 \
# --closs_weight 0.167236576878629 \
# --selection_criteria allsat_primary \
# --task sentiment \
# --num_samples 20 \
# --source_data 'laser_edit/data/sentiment/outputs.txt.init.jsonl' \
# --source_style 'negative' \
# --target_style 'positive' \
# --target_label_ids 1 1 \
# --thresholds 0.75 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint/' \
# --output_dir_prefix 'outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm'


# srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 10 \
# --beam_size 3 \
# --loss_weights 0.1 0.9 \
# --selection_criteria allsat_primary \
# --task sentiment \
# --num_samples 20 \
# --source_data 'laser_edit/data/sentiment/dev_set.jsonl' \
# --source_style 'positive' \
# --target_style 'negative' \
# --target_label_ids 1 0 \
# --thresholds 0.9 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint" \
# --tokenizer_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint" \
# --model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
# --output_dir_prefix 'outputs/sentiment/final' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm' \
# --dont_skip_allsat \
# --server_time_limit 12


# srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 10 \
# --beam_size 3 \
# --loss_weights 0.1 0.9 \
# --selection_criteria allsat_primary \
# --task sentiment \
# --num_samples 20 \
# --source_data 'laser_edit/data/sentiment/dev_set.jsonl' \
# --source_style 'positive' \
# --target_style 'negative' \
# --target_label_ids 0 0 \
# --thresholds 0.9 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint" \
# --tokenizer_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint" \
# --model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
# --output_dir_prefix 'outputs/sentiment/final' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm' \
# --dont_skip_allsat \
# --server_time_limit 12


srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 --num_edit_token_per_step 5 --locate_unit word --k_per_location 10 --n_iter 10 --beam_size 3 --loss_weights 0.1 0.9 --selection_criteria allsat_primary --task sentiment --num_samples 20 --source_data laser_edit/data/sentiment/dev_set.jsonl --source_style positive --target_style negative --target_label_ids 1 0 --thresholds 0.9 --wandb_project sentiment-decoding --model_paths gpt2-large /home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint --tokenizer_paths gpt2-large /home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint --model_types AutoModelForCausalLM AutoModelForSequenceClassification --output_dir_prefix outputs/sentiment/final --slurm_job_id 9051 --early_stopping_patience 0 --locate_method grad_norm --dont_skip_allsat --server_time_limit 12