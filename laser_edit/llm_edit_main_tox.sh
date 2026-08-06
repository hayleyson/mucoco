#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=edit_iter
#SBATCH --output='laser_edit/_slurm_outs/edit_iter_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache


export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/llm"

EXP_LABEL="nontoxic_masked"
TOTAL_ITERATION=1

INPUT_FILE_PATH="/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
ORIG_TEXT_PATH="/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
# LOCATED_RESULTS_FILE="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/processed_results/Qwen2.5-7B-Instruct_locate_toxic_5shot_type1_v3_baselm_gens_nontoxic_1782821550_processed.jsonl"

PRETRAINED_MODEL_PATH="/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" #"microsoft/Phi-3.5-mini-instruct"
PROMPT_TYPE="nontoxic_masked"
TASK="toxicity"
LABEL_ID=0
LOCATE_OPTION="grad_norm"
THRESHOLD=0.95
LOSS_NAME="classification_no_prefix_logprobloss"

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
--max_num_tokens 7
# --located_results_file $LOCATED_RESULTS_FILE

