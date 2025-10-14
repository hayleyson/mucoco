#!/bin/bash
#SBATCH --nodelist=n01
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=nli_energy
#SBATCH --output='new_module/_slurm_outs/nli_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache


# dataset : new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl
# dataset for sweep : new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105_sweep_1000.jsonl

# # 튜닝 다시..
# srun python new_module/new_mlm_reranking_all_sweep_n_iter.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 5 \
# --n_iter 10 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/data/hyeryung/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data 'new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105_sweep_1000.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --min_epsilons 0.99 \
# --wandb_project 'nli-decoding' \
# --model_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel

# # 확정된 hyperparameter로 돌리기 
# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 5 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/data/hyeryung/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data 'new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --min_epsilons 0.99 \
# --wandb_project 'nli-decoding' \
# --model_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel

# min epsilon에 대한 ablation
srun python new_module/new_mlm_reranking_all_sweep.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 5 \
--n_iter 1 \
--loss_weights 1 1 \
--selection_criteria allsat_primary \
--cache_dir '/data/hyeryung/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--task nli \
--output_dir_prefix 'outputs/nli/' \
--source_data 'new_module/data/logical-consistency/anli-r2-test_prompt_4.jsonl' \
--source_style 'inconsistent' \
--target_style 'consistent' \
--target_label_ids 1 1 \
--min_epsilons 0.99 \
--wandb_project 'nli-decoding' \
--model_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--locate_method 'grad_norm' \
--losses gpt2_no_prefix classification \
--model_types AutoModelForCausalLM EncoderModel

python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 5 \
--n_iter 1 \
--loss_weights 1 1 \
--selection_criteria allsat_primary \
--cache_dir '/data/hyeryung/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--task nli \
--output_dir_prefix 'outputs/nli/' \
--source_data 'new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
--source_style 'inconsistent' \
--target_style 'consistent' \
--target_label_ids 1 1 \
--min_epsilons 0.99 \
--wandb_project 'nli-decoding' \
--model_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--locate_method 'grad_norm' \
--losses gpt2_no_prefix classification bertscore\
--model_types AutoModelForCausalLM EncoderModel