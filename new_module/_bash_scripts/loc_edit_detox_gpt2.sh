#!/bin/bash
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=detox_energy
#SBATCH --output='new_module/_slurm_outs/detox_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache


# srun python new_module/ebm_edit_main_sweep.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7 \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task toxicity \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --source_data '/home/hyeryung/data/mucoco/new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --thresholds 0.95 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# srun python new_module/ebm_edit_main_sweep_n_iter.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7 \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 15 \
# --n_iter 10 \
# --loss_weights 1 10 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task toxicity \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --source_data '/home/hyeryung/data/mucoco/new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --thresholds 0.95 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# srun python new_module/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7 \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task toxicity \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --source_data '/home/hyeryung/data/mucoco/new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_9.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --thresholds 0.9 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# srun python new_module/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7 \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1 10 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task toxicity \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --source_data '/home/hyeryung/data/mucoco/new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_9.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --thresholds 0.9 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification


# gpt2가 생성한 것을 고치기
# hyperparameters: loss weights 1,10 / beam 5 / k 15 / n_iter 1
# srun python new_module/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7 \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1 10 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task toxicity \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --source_data 'new_module/data/toxicity-avoidance/testset_gpt2_2500_below_nontoxic_threshold_468.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --thresholds 0.95 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

srun python new_module/ebm_edit_main.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7 \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 15 \
--n_iter 1 \
--loss_weights 1 10 \
--selection_criteria allsat_primary \
--cache_dir '/home/hyeryung/data/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--task toxicity \
--output_dir_prefix 'outputs/toxicity/llm' \
--source_data 'new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--thresholds 0.95 \
--wandb_project 'toxicity-decoding' \
--model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--locate_method 'grad_norm' \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification