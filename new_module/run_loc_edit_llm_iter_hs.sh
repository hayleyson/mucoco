#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=edit_iter
#SBATCH --output='new_module/_slurm_outs/edit_iter_%j.out'
#SBATCH --nodelist=n02

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/.cache
export HF_DATASETS_CACHE=/data/hyeryung/.cache
export TRANSFORMERS_CACHE=/data/hyeryung/.cache
export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/data/hyeryung/mucoco/outputs/llmedit/results_2025"

EXP_LABEL="6_nli"
TOTAL_ITERATION=1

INPUT_FILE_PATH="/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"
ORIG_TEXT_PATH="/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"

# PRETRAINED_MODEL_PATH="/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/"
PRETRAINED_MODEL_PATH="/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" #"microsoft/Phi-3.5-mini-instruct"
PROMPT_TYPE="set_consistency_both"
TASK="set_nli"
# TASK="set_lconvqa"
LABEL_ID=1
LOCATE_OPTION="attention"
THRESHOLD=-1

# toxicity (target: nontoxic) - 0
# sentiment (target: positive) - 1
# sentiment (target: negative) - 0
# logical consistency (target: consistent) - 1
# formality transfer (target: formal) - 1
# formality transfer (target: informal) - 0

# 'nli_both', 
# 'nontoxic_both', 
# 'form_both', 'inform_both'
# 'senti_pos_both, 'senti_neg_both' 

srun python new_module/loc_edit_llm_iter_sc_v1.py \
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
--total_iteration $TOTAL_ITERATION


