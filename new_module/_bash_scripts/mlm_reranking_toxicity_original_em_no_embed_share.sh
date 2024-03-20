#!/bin/bash
#SBATCH --job-name=t_cb_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/t_cb_em_%j.out'
#SBATCH --partition=P1

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export LOGGING_LEVEL=INFO

srun python new_module/mlm_reranking_all.py --method mlm-reranking \
--num_edit_token_per_step 4  \
--locate_unit token \
--k_per_location 3 \
--n_iter 10 \
--closs_weight 0.9 \
--selection_criteria allsat_primary \
--task toxicity \
--num_samples 10 \
--source_data 'new_module/data/toxicity-avoidance/dev_set.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--min_epsilons 0.9 \
--wandb_project 'toxicity-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/models_re/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint/' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/models_re/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint/' \
--model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
--output_dir_prefix 'outputs/toxicity/devset' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--server_time_limit 12 \
--dont_skip_allsat
