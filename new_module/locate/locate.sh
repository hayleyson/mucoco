#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --output='new_module/_slurm_outs/loc_locate_edit_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

root_dir="/home/hyeryung/data"

MAX_NUM_TOKENS=7

# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
# --input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels_raveled.jsonl" \
# --output_file "new_module/locate/results/toxicspans/energy_model_gradient_norm_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task toxicity \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method grad_norm

# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
# --input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels_raveled.jsonl" \
# --output_file "new_module/locate/results/toxicspans/energy_model_attention_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task toxicity \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method attention

# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/" \
# --input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels_raveled.jsonl" \
# --output_file "new_module/locate/results/toxicspans/classifier_gradient_norm_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task toxicity \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method grad_norm

# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/" \
# --input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels_raveled.jsonl" \
# --output_file "new_module/locate/results/toxicspans/classifier_attention_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task toxicity \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method attention



# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/" \
# --input_file "new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final_raveled.jsonl" \
# --output_file "new_module/locate/results/inconsistentspans/energy_model_gradient_norm_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task nli \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method grad_norm

# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/" \
# --input_file "new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final_raveled.jsonl" \
# --output_file "new_module/locate/results/inconsistentspans/energy_model_attention_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task nli \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method attention


# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/" \
# --input_file "new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final_raveled.jsonl" \
# --output_file "new_module/locate/results/inconsistentspans/classifier_gradient_norm_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task nli \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method grad_norm

# srun python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "${root_dir}/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/" \
# --input_file "new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final_raveled.jsonl" \
# --output_file "new_module/locate/results/inconsistentspans/classifier_attention_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
# --task nli \
# --label_id 0 \
# --max_num_tokens $MAX_NUM_TOKENS \
# --locate_method attention


srun python new_module/locate/new_locate_utils.py \
--pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
--input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl" \
--output_file "new_module/locate/results/toxicspans_extended/energy_model_gradient_norm_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
--task toxicity_extended \
--label_id 0 \
--max_num_tokens $MAX_NUM_TOKENS \
--locate_method grad_norm

srun python new_module/locate/new_locate_utils.py \
--pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
--input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl" \
--output_file "new_module/locate/results/toxicspans_extended/energy_model_attention_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
--task toxicity_extended \
--label_id 0 \
--max_num_tokens $MAX_NUM_TOKENS \
--locate_method attention

srun python new_module/locate/new_locate_utils.py \
--pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/" \
--input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl" \
--output_file "new_module/locate/results/toxicspans_extended/classifier_gradient_norm_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
--task toxicity_extended \
--label_id 0 \
--max_num_tokens $MAX_NUM_TOKENS \
--locate_method grad_norm

srun python new_module/locate/new_locate_utils.py \
--pretrained_model_path "${root_dir}/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/" \
--input_file "new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl" \
--output_file "new_module/locate/results/toxicspans_extended/classifier_attention_max_num_tokens_${MAX_NUM_TOKENS}.jsonl" \
--task toxicity_extended \
--label_id 0 \
--max_num_tokens $MAX_NUM_TOKENS \
--locate_method attention