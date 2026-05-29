#!/bin/bash
#SBATCH --nodelist=n01
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=nli_energy
#SBATCH --output='laser_edit/_slurm_outs/nli_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 5 \
--n_iter 1 \
--loss_weights 1 1 \
--selection_criteria allsat_primary \
--cache_dir '/home/hyeryung/data/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task nli \
--output_dir_prefix 'outputs/nli/' \
--source_data 'laser_edit/data/logical-consistency/llama/filtered_0.99_r2-test_500_Llama-3.1-8B-Instruct_49578.jsonl' \
--source_style 'inconsistent' \
--target_style 'consistent' \
--target_label_ids 1 1 \
--thresholds 0.99 \
--wandb_project 'nli-decoding' \
--model_paths 'meta-llama/Llama-3.1-8B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--tokenizer_paths 'meta-llama/Llama-3.1-8B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--locate_method 'grad_norm' \
--losses gpt2_no_prefix classification \
--model_types AutoModelForCausalLM EncoderModel

# # sweep으로 돌릴 때 7,15 / 7,10 에서 에러가 나서 따로 돌림
# srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 1  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1.0 0.01 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data 'laser_edit/data/logical-consistency/llama/filtered_0.99_r2-test_500_Llama-3.1-8B-Instruct_49578.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --thresholds 0.99 \
# --wandb_project 'nli-decoding' \
# --model_paths 'meta-llama/Llama-3.1-8B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --tokenizer_paths 'meta-llama/Llama-3.1-8B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel

# # threshold 0.9 데이터에 대해서 실행
# srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 1  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 10 \
# --n_iter 1 \
# --loss_weights 1.0 0.1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data 'laser_edit/data/logical-consistency/llama/filtered_0.9_r2-test_500_Llama-3.1-8B-Instruct_49578.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --thresholds 0.9 \
# --wandb_project 'nli-decoding' \
# --model_paths 'meta-llama/Llama-3.1-8B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --tokenizer_paths 'meta-llama/Llama-3.1-8B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel
