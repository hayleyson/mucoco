#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=plain_nli
#SBATCH --output='laser_edit/_slurm_outs/plain_llm_nli_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

# Method: Plain LLM edit | Task: nli
# Adapted from llm_edit_baseline_main.sh + qwen3_editor_full_pro6000.sh (plain/nli)

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/home/hyeryung/data/mucoco/outputs/nli/llm"
EXP_LABEL="nli_notmasked"

INPUT_FILE_PATH="/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"
ORIG_TEXT_PATH="/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"
PRETRAINED_MODEL_PATH="/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PROMPT_TYPE="nli_notmasked"
TASK="nli"
LABEL_ID=1
THRESHOLD=0.99
LOSS_NAME="classification"

srun python laser_edit/llm_edit_baseline_main.py \
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
--threshold $THRESHOLD \
--loss_name $LOSS_NAME
