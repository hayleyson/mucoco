#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=laser_ebm_nli
#SBATCH --output='laser_edit/_slurm_outs/laser_ebm_nli_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: LASER & EBM edit | Task: nli (contradiction avoidance)
# Table 14: (w_f,w_c)=(1,1), N=1, m=3, l=7, k=5, n_b=5, epsilon=-log(0.99)
# Table 13: GradNorm localization, l=7

python laser_edit/ebm_edit_main.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7 \
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
--task nli \
--output_dir_prefix 'outputs/nli/ebm/' \
--source_data '/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
--source_style 'inconsistent' \
--target_style 'consistent' \
--target_label_ids 1 1 \
--thresholds 0.99 \
--wandb_project 'nli-decoding' \
--model_paths 'Qwen/Qwen2.5-7B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
--locate_method 'grad_norm' \
--losses gpt2_no_prefix classification \
--model_types AutoModelForCausalLM EncoderModel \
--dont_skip_allsat
