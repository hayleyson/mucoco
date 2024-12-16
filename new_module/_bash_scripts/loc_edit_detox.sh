#!/bin/bash
#SBATCH -J Serial_gpu_job
#SBATCH -p gpu-farm
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=detox_energy
#SBATCH --output='new_module/_slurm_outs/detox_decoding_%j.out'

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
# --task toxicity \
# --output_dir_prefix 'outputs/toxicity/llm' \
# --source_data '/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl' \
# --source_style 'toxic' \
# --target_style 'nontoxic' \
# --target_label_ids 0 0 \
# --min_epsilons 0.998671 \
# --wandb_project 'toxicity-decoding' \
# --model_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/' \
# --tokenizer_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification \
# --wandb_run_id p8srg772 \
# --resume

srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 10 \
--n_iter 1 \
--loss_weights 0.1 1.0 \
--selection_criteria allsat_primary \
--cache_dir '/data/hyeryung/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task toxicity \
--output_dir_prefix 'outputs/toxicity/llm' \
--source_data '/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--min_epsilons 0.95 \
--wandb_project 'toxicity-decoding' \
--model_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--tokenizer_paths 'google/gemma-2-2b' '/data/hyeryung/loc_edit/models/clean/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--locate_method 'grad_norm' \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification
