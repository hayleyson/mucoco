#!/bin/bash
#SBATCH --job-name=f_bv1_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/f_bv1_em_%j.out'
#SBATCH --partition=P2

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export LOGGING_LEVEL=INFO

srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v1 \
--num_edit_token_per_step 3  \
--locate_unit word \
--k_per_location 10 \
--n_iter 5 \
--loss_weights 0.1 1 \
--beam_size 5 \
--selection_criteria weighted_sum \
--task formality \
--source_data 'data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
--source_style 'informal' \
--target_style 'formal' \
--target_label_ids 1 1 \
--min_epsilons 0.99 \
--wandb_project 'formality-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
--model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
--output_dir_prefix 'outputs/formality/final' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--server_time_limit 12 \
--dont_skip_allsat \
--num_samples 20
