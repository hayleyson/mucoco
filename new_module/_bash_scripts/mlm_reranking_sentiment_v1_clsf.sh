#!/bin/bash
#SBATCH --job-name=s_bv1_clsf
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/s_bv1_clsf_%j.out'
#SBATCH --partition=P1

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v1 \
--num_edit_token_per_step 5  \
--locate_unit word \
--k_per_location 10 \
--n_iter 3 \
--closs_weight 0.43454090274164 \
--beam_size 3 \
--selection_criteria allsat_primary \
--task sentiment \
--num_samples 20 \
--source_data 'new_module/data/sentiment/outputs.txt.init.jsonl' \
--source_style 'negative' \
--target_style 'positive' \
--target_label_ids 1 1 \
--min_epsilons 0.75 \
--wandb_project 'sentiment-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds/step_114500_best_checkpoint' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds/step_114500_best_checkpoint/' \
--output_dir_prefix 'outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0
