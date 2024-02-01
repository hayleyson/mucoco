#!/bin/bash
#SBATCH --job-name=t_cb_clsf
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/t_cb_clsf_%j.out'
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
--task toxicity \
--num_samples 10 \
--source_data 'new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--min_epsilons 0.75 \
--wandb_project 'toxicity-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/step_2600_best_checkpoint/' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/step_2600_best_checkpoint//' \
--output_dir_prefix 'outputs/toxicity/mlm-reranking/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0
