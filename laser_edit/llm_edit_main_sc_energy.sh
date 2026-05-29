#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --job-name=edit_iter
#SBATCH --output='laser_edit/_slurm_outs/edit_iter_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache


export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID
DIRECTORY="outputs/sc_energy/set_nli/llm/testset_incon_300"

EXP_LABEL="set_nli_both_v1"
# EXP_LABEL="set_vqa_masked_v1"
# EXP_LABEL="set_vqa_masked_v2-1"
# EXP_LABEL="set_nli_masked_v2-1"
TOTAL_ITERATION=8

INPUT_FILE_PATH="laser_edit/data/set_nli/testset_incon_300/set_nli_testset_incon_300.jsonl"

PRETRAINED_MODEL_PATH="/home/hyeryung/data/mucoco/laser_edit/set_consistency_energy/params_set_nli.yaml"

LLM_NAME="Qwen/Qwen3-8B" #"microsoft/Phi-3.5-mini-instruct"
PROMPT_TYPE="set_consistency_both"
TASK="set_nli"
# TASK="set_lconvqa"
LABEL_ID=1
LOCATE_OPTION="grad_norm" # just a placeholder
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

srun -n 1 -c 1 python laser_edit/llm_edit_main_sc_energy.py \
$JOB_ID \
--exp_label $EXP_LABEL \
--directory $DIRECTORY \
--input_file_path $INPUT_FILE_PATH \
--pretrained_model_path $PRETRAINED_MODEL_PATH \
--llm_name $LLM_NAME \
--prompt_type $PROMPT_TYPE \
--task $TASK \
--label_id $LABEL_ID \
--locate_option $LOCATE_OPTION \
--threshold $THRESHOLD \
--total_iteration $TOTAL_ITERATION \
--use_incon_samples \
--n_samples 300 \
--losses sc_energy \
--max_num_tokens 7 \
--use_vllm
