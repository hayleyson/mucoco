#!/bin/bash
#SBATCH --job-name=t_bv2_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/t_bv2_em_%j.out'
#SBATCH --partition=P1


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v2 \
--num_edit_token_per_step 5  \
--locate_unit word \
--n_iter 3 \
--closs_weight 0.322924626233093 \
--beam_size 5 \
--selection_criteria allsat_primary \
--task toxicity \
--num_samples 10 \
--source_data 'new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--min_epsilons 0.75 \
--wandb_project 'toxicity-decoding' \
--model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint/' \
--tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint//' \
--output_dir_prefix 'outputs/toxicity/mlm-reranking/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0
