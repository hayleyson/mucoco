#!/bin/bash
#SBATCH --job-name=f_cb_clsf
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/f_cb_clsf_%j.out'
#SBATCH --partition=P1

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python new_module/mlm_reranking_all.py --method mlm-reranking \
--num_edit_token_per_step 4  \
--locate_unit token \
--k_per_location 3 \
--n_iter 2 \
--closs_weight 0.5 \
--selection_criteria allsat_primary \
--task formality \
--source_data 'data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
--source_style 'informal' \
--target_style 'formal' \
--target_label_ids 1 1 \
--min_epsilons 0.75 \
--wandb_project 'formality-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds/step_1116_best_checkpoint/' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds/step_1116_best_checkpoint/' \
--output_dir_prefix 'outputs/formality/mlm-reranking/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0

# srun python new_module/mlm_reranking_all.py --method mlm-reranking \
# --num_edit_token_per_step 4  \
# --locate_unit token \
# --k_per_location 3 \
# --n_iter 2 \
# --closs_weight 0.5 \
# --selection_criteria allsat_primary \
# --task formality \
# --source_data 'data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.75 \
# --model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/step_22500_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/step_22500_best_checkpoint/' \
# --output_dir_prefix 'outputs/formality/mlm-reranking/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/'
