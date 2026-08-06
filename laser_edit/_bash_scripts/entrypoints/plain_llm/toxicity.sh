#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=plain_tox
#SBATCH --output='laser_edit/_slurm_outs/plain_llm_tox_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

# Method: Plain LLM edit | Task: toxicity
# Source: llm_edit_baseline_main.sh (active toxicity config)

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/llm"
EXP_LABEL="toxicity_notmasked"

INPUT_FILE_PATH="/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
ORIG_TEXT_PATH="/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
PRETRAINED_MODEL_PATH="/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PROMPT_TYPE="nontoxic_notmasked"
TASK="toxicity"
LABEL_ID=0
THRESHOLD=0.95
LOSS_NAME="classification_no_prefix_logprobloss"

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
