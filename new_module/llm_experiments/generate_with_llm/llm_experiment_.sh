#!/bin/bash
#SBATCH --job-name=t_bv0_clsf
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/t_bv0_clsf_%j.out'

source /home/hyeryung/.bashrc
source /home/hyeryung/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache
export LOGGING_LEVEL=INFO

# srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 3 \
# --closs_weight 0.167236576878629 \
# --selection_criteria allsat_primary \
# --task toxicity \
# --num_samples 10 \
# --source_data 'new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --min_epsilons 0.75 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint/' \
# --output_dir_prefix 'outputs/toxicity/mlm-reranking/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm'

# python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --beam_size 3 \
# --loss_weights 0.1 1.0 \
# --k_per_location 5 \
# --n_iter 10 \
# --selection_criteria allsat_primary \
# --task toxicity \
# --num_samples 10 \
# --source_data '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --min_epsilons 0.9 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'meta-llama/Meta-Llama-3-8B' '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --tokenizer_paths 'meta-llama/Meta-Llama-3-8B' '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm' \
# --server_time_limit 12 \
# --device 'cuda' \
# --cache_dir '/data/hyeryung/hf_cache' \
# --model_types 'AutoModelForCausalLM' 'AutoModelForSequenceClassification'

# python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --beam_size 3 \
# --loss_weights 0.1 1.0 \
# --k_per_location 5 \
# --n_iter 10 \
# --selection_criteria allsat_primary \
# --task toxicity \
# --num_samples 10 \
# --source_data '/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --min_epsilons 0.9 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'Qwen/Qwen2.5-7B' '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --tokenizer_paths 'Qwen/Qwen2.5-7B' '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm' \
# --dont_skip_allsat \
# --server_time_limit 12 \
# --device 'cuda' \
# --cache_dir '/data/hyeryung/hf_cache' \
# --model_types 'AutoModelForCausalLM' 'AutoModelForSequenceClassification'

python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 5  \
--locate_unit word \
--beam_size 3 \
--loss_weights 0.1 1.0 \
--k_per_location 5 \
--n_iter 10 \
--selection_criteria allsat_primary \
--task toxicity \
--num_samples 10 \
--source_data '/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_noprompt_150.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--min_epsilons 0.9 \
--wandb_project 'toxicity-decoding' \
--model_paths 'Qwen/Qwen2.5-7B' '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--tokenizer_paths 'Qwen/Qwen2.5-7B' '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--output_dir_prefix 'outputs/toxicity/llm' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--server_time_limit 12 \
--dont_skip_allsat \
--device 'cuda' \
--max_tokens_per_span 3 \
--cache_dir '/data/hyeryung/hf_cache' \
--model_types 'AutoModelForCausalLM' 'AutoModelForSequenceClassification'

# python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --beam_size 3 \
# --loss_weights 0.1 1.0 \
# --k_per_location 5 \
# --n_iter 10 \
# --selection_criteria allsat_primary \
# --task sentiment \
# --num_samples 10 \
# --source_data '/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/sentiment/gpt-3.5-turbo-0125_pplm_prompts_noprompt_150.jsonl' \
# --source_style 'nan' \
# --target_style 'negative' \
# --target_label_ids 0 0 \
# --min_epsilons 0.9 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'Qwen/Qwen2.5-7B' 'siebert/sentiment-roberta-large-english' \
# --tokenizer_paths 'Qwen/Qwen2.5-7B' 'siebert/sentiment-roberta-large-english' \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm' \
# --server_time_limit 12 \
# --device 'cuda' \
# --cache_dir '/data/hyeryung/hf_cache' \
# --model_types 'AutoModelForCausalLM' 'AutoModelForSequenceClassification'

# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 100  \
# --locate_unit word \
# --beam_size 3 \
# --loss_weights 0.1 1.0 \
# --k_per_location 5 \
# --n_iter 10 \
# --selection_criteria allsat_primary \
# --task sentiment \
# --num_samples 10 \
# --source_data '/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/sentiment/gpt-3.5-turbo-0125_pplm_prompts_noprompt_150.jsonl' \
# --source_style 'nan' \
# --target_style 'negative' \
# --target_label_ids 0 0 \
# --min_epsilons 0.9 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'Qwen/Qwen2.5-7B' 'siebert/sentiment-roberta-large-english' \
# --tokenizer_paths 'Qwen/Qwen2.5-7B' 'siebert/sentiment-roberta-large-english' \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method 'grad_norm' \
# --server_time_limit 12 \
# --device 'cuda' \
# --cache_dir '/data/hyeryung/hf_cache' \
# --model_types 'AutoModelForCausalLM' 'AutoModelForSequenceClassification'