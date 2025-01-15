#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=edit_iter
#SBATCH --output='new_module/_slurm_outs/edit_iter_%j.out'
#SBATCH --nodelist=n01

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/.cache
export HF_DATASETS_CACHE=/data/hyeryung/.cache
export TRANSFORMERS_CACHE=/data/hyeryung/.cache
export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/data/hyeryung/mucoco/new_module/iter_loc_edit_qwen"

EXP_LABEL="31_neg"
TOTAL_ITERATION=2

INPUT_FILE_PATH="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl"
ORIG_TEXT_PATH="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl"

PRETRAINED_MODEL_PATH="/data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" #"microsoft/Phi-3.5-mini-instruct"
PROMPT_TYPE="senti_neg_both"
TASK="sentiment"
LABEL_ID=0
LOCATE_OPTION="attention"
THRESHOLD=0.92

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
--total_iteration $TOTAL_ITERATION


