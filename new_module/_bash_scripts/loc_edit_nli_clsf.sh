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
export HF_HOME=/home/hyeryung/data/hf_cache

# # clsf에 대해서 loss weights tuning -> k, beam tuning
# srun python new_module/new_mlm_reranking_all_sweep.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 5 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data '/home/hyeryung/data/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105_sweep_1000.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --thresholds 0.98974 \
# --wandb_project 'nli-decoding' \
# --model_paths 'Qwen/Qwen2.5-7B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/' \
# --tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel

# # 전체 데이터에 대해서 돌리기
# # 250302 : num_edit_token_per_step = 1로도 돌려봄
srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
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
--source_data '/home/hyeryung/data/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
--source_style 'inconsistent' \
--target_style 'consistent' \
--target_label_ids 1 1 \
--thresholds 0.98974 \
--wandb_project 'nli-decoding' \
--model_paths 'Qwen/Qwen2.5-7B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/' \
--tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/' \
--locate_method 'grad_norm' \
--losses gpt2_no_prefix classification \
--model_types AutoModelForCausalLM EncoderModel