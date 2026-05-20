#!/bin/bash
#SBATCH -J Serial_gpu_job
#SBATCH -p gpu-farm
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=senti_energy
#SBATCH --output='new_module/_slurm_outs/positive_decoding_%j.out'

module purge module load cuda/12.1
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 10 \
# --loss_weights 0.1 1.0 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task sentiment \
# --output_dir_prefix 'outputs/sentiment/positive_gemma/' \
# --source_data '/home/hyeryung/data/mucoco/new_module/data/sentiment/gemma_pplm_below_positive_threshold.jsonl' \
# --source_style 'negative' \
# --target_style 'positive' \
# --target_label_ids 1 1 \
# --thresholds 0.97 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'google/gemma-2-2b' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
# --tokenizer_paths 'google/gemma-2-2b' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification \
# --wandb_run_id ba2h70jd \
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
--cache_dir '/home/hyeryung/data/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task sentiment \
--output_dir_prefix 'outputs/sentiment/positive_gemma/' \
--source_data '/home/hyeryung/data/mucoco/new_module/data/sentiment/gemma_pplm_below_positive_threshold.jsonl' \
--source_style 'negative' \
--target_style 'positive' \
--target_label_ids 1 1 \
--thresholds 0.9999994 \
--wandb_project 'sentiment-decoding' \
--model_paths 'google/gemma-2-2b' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier/step_83000' \
--tokenizer_paths 'google/gemma-2-2b' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier/step_83000' \
--locate_method 'grad_norm' \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification \
--wandb_run_id 5jy4oljq \
--resume

