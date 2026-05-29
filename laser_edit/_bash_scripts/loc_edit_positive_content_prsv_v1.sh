#!/bin/bash
#SBATCH --nodelist=n02
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=senti_energy
#SBATCH --output='laser_edit/_slurm_outs/positive_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

# datasets
# - new dev set : /home/hyeryung/data/mucoco/laser_edit/data/sentiment/new_dev_set_below_positive_threshold_200.jsonl
# - dev set (used as test set): /home/hyeryung/data/mucoco/laser_edit/data/sentiment/dev_set_below_positive_threshold_778.jsonl


# srun python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1 1 1 1 \
# --selection_criteria allsat_rest \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task sentiment \
# --output_dir_prefix 'outputs/sentiment/positive_gpt2/' \
# --source_data '/home/hyeryung/data/mucoco/laser_edit/data/sentiment/new_dev_set_below_positive_threshold_200.jsonl' \
# --source_style 'negative' \
# --target_style 'positive' \
# --target_label_ids 1 1 1 1 \
# --thresholds 0.97 \
# --wandb_project 'sentiment-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss bertscore edit_distance \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification \
# --memo "add least, proba scale (updated)"



srun python laser_edit/ebm_edit_main_sweep.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 7 \
--k_per_location 15 \
--n_iter 1 \
--loss_weights 1 1 1 1 \
--selection_criteria allsat_rest \
--cache_dir '/home/hyeryung/data/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task sentiment \
--output_dir_prefix 'outputs/sentiment/positive_gpt2/' \
--source_data '/home/hyeryung/data/mucoco/laser_edit/data/sentiment/new_dev_set_below_positive_threshold_200.jsonl' \
--source_style 'negative' \
--target_style 'positive' \
--target_label_ids 1 1 1 1 \
--thresholds 0.97 \
--wandb_project 'sentiment-decoding' \
--model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
--tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint' \
--locate_method 'grad_norm' \
--losses gpt2 classification_no_prefix_logprobloss bertscore edit_distance \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification \
--memo "add least, log scale (final), sweep"
