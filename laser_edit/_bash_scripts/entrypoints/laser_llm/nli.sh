#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=laser_llm_nli
#SBATCH --output='laser_edit/_slurm_outs/laser_llm_nli_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

# Method: LASER & LLM edit | Task: nli
# Adapted from llm_edit_main.sh + qwen3_editor (laser/nli: prompt nli_both, no located file)

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/home/hyeryung/data/mucoco/outputs/nli/llm"
EXP_LABEL="nli_both"
TOTAL_ITERATION=1

INPUT_FILE_PATH="/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"
ORIG_TEXT_PATH="/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"

PRETRAINED_MODEL_PATH="/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/"
HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PROMPT_TYPE="nli_both"
TASK="nli"
LABEL_ID=1
LOCATE_OPTION="grad_norm"
THRESHOLD=0.99
LOSS_NAME="classification"
# Table 13 / §B.4: l=7 for contradiction avoidance (LaSEr-LLM Edit)
MAX_NUM_TOKENS=7

srun python laser_edit/llm_edit_main.py \
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
--max_num_tokens $MAX_NUM_TOKENS
