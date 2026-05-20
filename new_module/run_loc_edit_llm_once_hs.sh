#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=edit_once
#SBATCH --output='new_module/_slurm_outs/edit_once_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export HF_DATASETS_CACHE=/home/hyeryung/data/.cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID
DIRECTORY="outputs/nli_toxicity/llm"

EXP_LABEL="nli_toxicity_notmasked"

INPUT_FILE_PATH="/home/hyeryung/data/mucoco/new_module/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl"
ORIG_TEXT_PATH="/home/hyeryung/data/mucoco/new_module/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl"

# Order matches TASK=nli_toxicity -> locate_modes / eval_tasks: nli, then toxicity
PRETRAINED_MODEL_PATH_NLI="/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/"
PRETRAINED_MODEL_PATH_TOX="/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint"

HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" #"microsoft/Phi-3.5-mini-instruct"
PROMPT_TYPE="nli_toxicity_notmasked"
TASK="nli_toxicity"
# label_id / threshold / loss_name: one per --pretrained_model_path (same order)
LABEL_ID_NLI=1
LABEL_ID_TOX=0
THRESHOLD_NLI=0.99
THRESHOLD_TOX=0.95
LOSS_NAME_NLI="classification"
LOSS_NAME_TOX="classification_no_prefix_logprobloss"
LOCATE_OPTION="grad_norm"

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

srun -n 1 -c 1 python new_module/loc_edit_llm_once.py \
$JOB_ID \
--exp_label $EXP_LABEL \
--directory $DIRECTORY \
--input_file_path $INPUT_FILE_PATH \
--orig_text_path $ORIG_TEXT_PATH \
--pretrained_model_path $PRETRAINED_MODEL_PATH_NLI $PRETRAINED_MODEL_PATH_TOX \
--hf_model_name $HF_MODEL_NAME \
--prompt_type $PROMPT_TYPE \
--task $TASK \
--label_id $LABEL_ID_NLI $LABEL_ID_TOX \
--locate_option $LOCATE_OPTION \
--threshold $THRESHOLD_NLI $THRESHOLD_TOX \
--loss_name $LOSS_NAME_NLI $LOSS_NAME_TOX \
--use_vllm
