#!/bin/bash
#SBATCH --job-name=s_bv1_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/s_bv1_em_%j.out'
#SBATCH --nodelist=n02

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache
export LOGGING_LEVEL=INFO

srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v1 \
--num_edit_token_per_step 5  \
--locate_unit word \
--k_per_location 10 \
--n_iter 3 \
--loss_weights 0.1 0.7 0.2 \
--beam_size 3 \
--selection_criteria allsat_primary \
--task sentiment \
--num_samples 20 \
--source_data 'new_module/data/sentiment/dev_set.jsonl' \
--source_style 'negative' \
--target_style 'positive' \
--target_label_ids 1 1 0 \
--min_epsilons 0.9 0.75 \
--wandb_project 'sentiment-decoding' \
--model_paths 'gpt2-large' '/data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' 'cointegrated/roberta-large-cola-krishna2020' \
--tokenizer_paths 'gpt2-large' '/data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' 'cointegrated/roberta-large-cola-krishna2020' \
--output_dir_prefix 'outputs/sentiment/devset/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--losses "gpt2" "classification_no_prefix_logprobloss" "classification_logprobloss" \
--dont_skip_allsat \
--model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" "AutoModelForSequenceClassification"

