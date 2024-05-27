#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodelist=n01
#SBATCH --gres=gpu:1
#SBATCH --job-name=formal_decode
#SBATCH --output='new_module/_slurm_outs/formal_decode_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v1 \
--num_edit_token_per_step 5  \
--locate_unit word \
--k_per_location 10 \
--n_iter 10 \
--loss_weights 0.1 1.0 \
--beam_size 3 \
--selection_criteria allsat_primary \
--task formality \
--source_data 'data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
--source_style 'informal' \
--target_style 'formal' \
--target_label_ids 1 1 \
--min_epsilons 0.99 \
--wandb_project 'formality-decoding' \
--model_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-pt16-formality-classifier-gyafc/step_2500_best_checkpoint/" \
--tokenizer_paths 'gpt2-large' "${DATA_DIR}/loc_edit/models/roberta-base-pt16-formality-classifier-gyafc/step_2500_best_checkpoint/" \
--model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
--output_dir_prefix 'outputs/formality/final' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--server_time_limit 12 \
--dont_skip_allsat \
--num_samples 20