#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=edit_once
#SBATCH --output='new_module/_slurm_outs/edit_once_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export HF_DATASETS_CACHE=/home/hyeryung/data/.cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/home/hyeryung/data/mucoco/outputs/llmedit/results_2025"
HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" #"microsoft/Phi-3.5-mini-instruct"

# 'nli_notmasked'
# 'nontoxic_notmasked'
#  'form_notmasked'
# 'senti_pos_notmasked'
# 'senti_neg_notmasked'
# 'inform_notmasked'

# toxicity (target: nontoxic) - 0
# sentiment (target: positive) - 1
# sentiment (target: negative) - 0
# logical consistency (target: consistent) - 1
# formality transfer (target: formal) - 1
# formality transfer (target: informal) - 0

###############################################################################################
# tox
###############################################################################################

INPUT_FILE_PATH="/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
ORIG_TEXT_PATH="/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"

PRETRAINED_MODEL_PATH="/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" #"microsoft/Phi-3.5-mini-instruct"
TASK="toxicity"
LABEL_ID=0
LOCATE_OPTION="grad_norm"
THRESHOLD=0.95
LOSS_NAME="classification_no_prefix_logprobloss"


# ########################################################
# # 5_tox
# ########################################################


EXP_LABEL="5_tox"
PROMPT_TYPE="nontoxic_notmasked"

srun python new_module/loc_edit_llm_once.py \
$JOB_ID \
--exp_label $EXP_LABEL \
--directory $DIRECTORY \
--input_file_path $INPUT_FILE_PATH \
--orig_text_path $ORIG_TEXT_PATH \
--pretrained_model_path $PRETRAINED_MODEL_PATH \
--hf_model_name $HF_MODEL_NAME \
--prompt_type $PROMPT_TYPE \
--task $TASK \
--label_id $LABEL_ID \
--locate_option $LOCATE_OPTION \
--threshold $THRESHOLD \
--loss_name $LOSS_NAME


########################################################
# 2_tox
########################################################

EXP_LABEL="2_tox"
TOTAL_ITERATION=1
PROMPT_TYPE="nontoxic_both"

srun python new_module/loc_edit_llm_iter.py \
$JOB_ID \
--exp_label $EXP_LABEL \
--directory $DIRECTORY \
--input_file_path $INPUT_FILE_PATH \
--orig_text_path $ORIG_TEXT_PATH \
--pretrained_model_path $PRETRAINED_MODEL_PATH \
--hf_model_name $HF_MODEL_NAME \
--prompt_type $PROMPT_TYPE \
--task $TASK \
--label_id $LABEL_ID \
--locate_option $LOCATE_OPTION \
--threshold $THRESHOLD \
--loss_name $LOSS_NAME \
--total_iteration $TOTAL_ITERATION \
--max_num_tokens 1


# ########################################################
# # 2_2_tox
# ########################################################

EXP_LABEL="2_2_tox"
TOTAL_ITERATION=1
PROMPT_TYPE="nontoxic_masked"

srun python new_module/loc_edit_llm_iter.py \
$JOB_ID \
--exp_label $EXP_LABEL \
--directory $DIRECTORY \
--input_file_path $INPUT_FILE_PATH \
--orig_text_path $ORIG_TEXT_PATH \
--pretrained_model_path $PRETRAINED_MODEL_PATH \
--hf_model_name $HF_MODEL_NAME \
--prompt_type $PROMPT_TYPE \
--task $TASK \
--label_id $LABEL_ID \
--locate_option $LOCATE_OPTION \
--threshold $THRESHOLD \
--loss_name $LOSS_NAME \
--total_iteration $TOTAL_ITERATION \
--max_num_tokens 1


# ###############################################################################################
# # nli
# ###############################################################################################

# INPUT_FILE_PATH="/home/hyeryung/data/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"
# ORIG_TEXT_PATH="/home/hyeryung/data/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"

# PRETRAINED_MODEL_PATH="/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/"


# TASK="nli"
# LABEL_ID=1
# LOCATE_OPTION="grad_norm"
# THRESHOLD=0.99
# LOSS_NAME="classification"


# ########################################################
# # 9_nli
# ########################################################

# EXP_LABEL="9_nli"
# PROMPT_TYPE="nli_notmasked"

# srun python new_module/loc_edit_llm_once.py \
# $JOB_ID \
# --exp_label $EXP_LABEL \
# --directory $DIRECTORY \
# --input_file_path $INPUT_FILE_PATH \
# --orig_text_path $ORIG_TEXT_PATH \
# --pretrained_model_path $PRETRAINED_MODEL_PATH \
# --hf_model_name $HF_MODEL_NAME \
# --prompt_type $PROMPT_TYPE \
# --task $TASK \
# --label_id $LABEL_ID \
# --locate_option $LOCATE_OPTION \
# --threshold $THRESHOLD \
# --loss_name $LOSS_NAME

# ########################################################
# # 6_nli
# ########################################################


# EXP_LABEL="6_nli"
# TOTAL_ITERATION=1
# PROMPT_TYPE="nli_both"

# srun python new_module/loc_edit_llm_iter.py \
# $JOB_ID \
# --exp_label $EXP_LABEL \
# --directory $DIRECTORY \
# --input_file_path $INPUT_FILE_PATH \
# --orig_text_path $ORIG_TEXT_PATH \
# --pretrained_model_path $PRETRAINED_MODEL_PATH \
# --hf_model_name $HF_MODEL_NAME \
# --prompt_type $PROMPT_TYPE \
# --task $TASK \
# --label_id $LABEL_ID \
# --locate_option $LOCATE_OPTION \
# --threshold $THRESHOLD \
# --loss_name $LOSS_NAME \
# --total_iteration $TOTAL_ITERATION \
# --max_num_tokens 1


# ########################################################
# # 6_2_nli
# ########################################################


# EXP_LABEL="6_2_nli"
# PROMPT_TYPE="nli_masked"

# srun python new_module/loc_edit_llm_iter.py \
# $JOB_ID \
# --exp_label $EXP_LABEL \
# --directory $DIRECTORY \
# --input_file_path $INPUT_FILE_PATH \
# --orig_text_path $ORIG_TEXT_PATH \
# --pretrained_model_path $PRETRAINED_MODEL_PATH \
# --hf_model_name $HF_MODEL_NAME \
# --prompt_type $PROMPT_TYPE \
# --task $TASK \
# --label_id $LABEL_ID \
# --locate_option $LOCATE_OPTION \
# --threshold $THRESHOLD \
# --loss_name $LOSS_NAME \
# --total_iteration $TOTAL_ITERATION \
# --max_num_tokens 1