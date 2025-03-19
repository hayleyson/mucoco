#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=edit_once
#SBATCH --output='/data3/saeheeeom/set_consistency/mucoco/new_module/_slurm_outs/edit_once_%j.out'
#SBATCH --nodelist=n02

source /data3/saeheeeom/.bashrc
source /data3/saeheeeom/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data3/saeheeeom/.cache
export HF_DATASETS_CACHE=/data3/saeheeeom/.cache
export TRANSFORMERS_CACHE=/data3/saeheeeom/.cache
export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID
DIRECTORY="/data3/saeheeeom/set_consistency/mucoco/new_module/iter_loc_edit_qwen"

EXP_LABEL="12_form"

INPUT_FILE_PATH="/data3/saeheeeom/set_consistency/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.jsonl"
ORIG_TEXT_PATH="/data3/saeheeeom/set_consistency/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.jsonl"

PRETRAINED_MODEL_PATH="/data3/saeheeeom/set_consistency/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" #"microsoft/Phi-3.5-mini-instruct"
PROMPT_TYPE="form_notmasked"
TASK="formality"
LABEL_ID=1
LOCATE_OPTION="grad_norm"
THRESHOLD=0.74

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

srun python /home/saeheeeom/data/set_consistency/mucoco/new_module/loc_edit_llm_once.py \
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
--threshold $THRESHOLD 

