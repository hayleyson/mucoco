#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --job-name=nli_toxicity_ebm
#SBATCH --output='new_module/_slurm_outs/nli_toxicity_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun python new_module/ebm_edit_main.py \
--method mlm-beamsearch-v0 \
--num_edit_token_per_step 3 \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 10 \
--n_iter 1 \
--loss_weights 1 10 1 \
--selection_criteria allsat_primary \
--cache_dir /home/hyeryung/data/hf_cache \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task nli_toxicity \
--output_dir_prefix outputs/nli_toxicity/ebm \
--source_data /home/hyeryung/data/mucoco/new_module/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl \
--source_style toxic \
--target_style nontoxic \
--target_label_ids 0 1 0 \
--thresholds 0.95 0.99 \
--threshold_scales probability probability \
--wandb_project nli-toxicity-decoding \
--model_paths Qwen/Qwen2.5-7B-Instruct /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/ /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
--tokenizer_paths Qwen/Qwen2.5-7B-Instruct /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/ /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
--locate_method grad_norm \
--losses gpt2_no_prefix classification classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM EncoderModel AutoModelForSequenceClassification