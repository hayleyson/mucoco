#!/bin/bash
#SBATCH --job-name=f_cb_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/f_cb_em_%j.out'
#SBATCH --nodelist=n01

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit


DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun python new_module/new_mlm_reranking_all.py --method mlm-reranking \
--num_edit_token_per_step 4  \
--locate_unit token \
--k_per_location 3 \
--n_iter 10 \
--loss_weights 0.1 0.9 \
--selection_criteria allsat_primary \
--task formality \
--source_data 'data/formality/GYAFC_Corpus/Entertainment_Music/test/formal' \
--source_style 'formal' \
--target_style 'informal' \
--target_label_ids 0 0 \
--min_epsilons 0.9 \
--num_samples 20 \
--wandb_project 'formality-decoding' \
--model_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/" \
--tokenizer_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/" \
--model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
--output_dir_prefix 'outputs/formality/final-informal' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--dont_skip_allsat \
--server_time_limit 12
