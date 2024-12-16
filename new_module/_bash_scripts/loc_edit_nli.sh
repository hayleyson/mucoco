#!/bin/bash
#SBATCH -J Serial_gpu_job
#SBATCH -p gpu-farm
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=nli_energy
#SBATCH --output='new_module/_slurm_outs/nli_decoding_%j.out'

module purge module load cuda/12.1
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 10 \
# --loss_weights 0.1 1.0 \
# --selection_criteria allsat_primary \
# --cache_dir '/data/hyeryung/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data '/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --min_epsilons 0.98974 \
# --wandb_project 'nli-decoding' \
# --model_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/' \
# --tokenizer_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel \
# --wandb_run_id jyfq8dca \
# --resume

# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 10 \
# --loss_weights 0.1 1.0 \
# --selection_criteria allsat_primary \
# --cache_dir '/data/hyeryung/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data '/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --min_epsilons 0.99 \
# --wandb_project 'nli-decoding' \
# --model_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --tokenizer_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel \
# --wandb_run_id gx5t4vva \
# --resume

srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 10 \
--n_iter 10 \
--loss_weights 0.1 1.0 \
--selection_criteria allsat_primary \
--cache_dir '/data/hyeryung/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task nli \
--output_dir_prefix 'outputs/nli/' \
--source_data '/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
--source_style 'inconsistent' \
--target_style 'consistent' \
--target_label_ids 1 1 \
--min_epsilons 0.99 \
--wandb_project 'nli-decoding' \
--model_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--tokenizer_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--locate_method 'attention' \
--losses gpt2_no_prefix classification \
--model_types AutoModelForCausalLM EncoderModel
