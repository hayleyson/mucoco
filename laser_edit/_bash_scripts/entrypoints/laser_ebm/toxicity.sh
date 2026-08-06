#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=laser_ebm_tox
#SBATCH --output='laser_edit/_slurm_outs/laser_ebm_tox_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: LASER & EBM edit | Task: toxicity
# Table 14: (w_f,w_c)=(1,10), N=1, m=3, l=7, k=10, n_b=5, epsilon=-log(0.95)
# Table 13: GradNorm localization, l=7

srun python laser_edit/ebm_edit_main.py \
--method mlm-beamsearch-v0 \
--num_edit_token_per_step 7 \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 10 \
--n_iter 1 \
--loss_weights 1 10 \
--selection_criteria allsat_primary \
--cache_dir /home/hyeryung/data/hf_cache \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task toxicity \
--output_dir_prefix outputs/toxicity/gpt3_5_gen/ebm \
--source_data /home/hyeryung/data/mucoco/laser_edit/data/toxicity-avoidance/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl \
--source_style toxic \
--target_style nontoxic \
--target_label_ids 0 0 \
--thresholds 0.95 \
--wandb_project toxicity-decoding \
--model_paths Qwen/Qwen2.5-7B-Instruct /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
--tokenizer_paths Qwen/Qwen2.5-7B-Instruct /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
--locate_method grad_norm \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification
