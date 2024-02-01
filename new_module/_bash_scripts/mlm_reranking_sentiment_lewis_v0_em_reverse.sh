#!/bin/bash
#SBATCH --job-name=sl_bv0_em_r
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/sl_bv0_em_r_%j.out'
#SBATCH --partition=P1

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit2

srun python new_module/mlm_reranking_all_lewis.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 5  \
--locate_unit word \
--k_per_location 10 \
--n_iter 3 \
--closs_weight 0.167236576878629 \
--selection_criteria allsat_primary \
--task sentiment-lewis-compr \
--num_samples 1 \
--source_data 'data/sentiment/yelp_li2018/sentiment.test.1' \
--source_style 'positive' \
--target_style 'negative' \
--target_label_ids 0 0 \
--min_epsilons 0.75 \
--wandb_project 'sentiment-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint/' \
--output_dir_prefix 'outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/lewis-compr/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0
