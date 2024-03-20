#!/bin/bash
#SBATCH --job-name=s_bv2_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/s_bv2_em_%j.out'
#SBATCH --partition=P1

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export LOGGING_LEVEL=INFO


srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v2 \
--num_edit_token_per_step 5  \
--locate_unit word \
--n_iter 10 \
--closs_weight 0.9 \
--beam_size 5 \
--selection_criteria allsat_primary \
--task sentiment \
--num_samples 20 \
--source_data 'new_module/data/sentiment/dev_set.jsonl' \
--source_style 'negative' \
--target_style 'positive' \
--target_label_ids 1 1 \
--min_epsilons 0.9 \
--wandb_project 'sentiment-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
--model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
--output_dir_prefix 'outputs/sentiment/devset' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--dont_skip_allsat \
--server_time_limit 12

