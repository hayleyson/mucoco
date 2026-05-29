#!/bin/bash
#SBATCH --nodelist=n01
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=formal_attention
#SBATCH --output='laser_edit/_slurm_outs/formality_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 7 \
--k_per_location 15 \
--n_iter 1 \
--loss_weights 1 1 \
--selection_criteria allsat_primary \
--cache_dir '/home/hyeryung/data/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task formality \
--output_dir_prefix 'outputs/formality/formal/' \
--source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
--source_style 'informal' \
--target_style 'formal' \
--target_label_ids 1 1 \
--thresholds 0.74 \
--wandb_project 'formality-decoding' \
--model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
--tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
--locate_method 'attention' \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification

