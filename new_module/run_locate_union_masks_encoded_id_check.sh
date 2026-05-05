#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=union_mask_chk
#SBATCH --output=new_module/_slurm_outs/run_locate_union_masks_encoded_id_check_%j.out

set -euo pipefail

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

MUCOCO_ROOT="${MUCOCO_ROOT:-/home/hyeryung/data/mucoco}"
cd "$MUCOCO_ROOT"

mkdir -p new_module/_slurm_outs
mkdir -p new_module/_outputs

export PYTHONPATH=.
DATA_DIR="${DATA_DIR:-/home/hyeryung/data}"
export HF_HOME="${HF_HOME:-$DATA_DIR/hf_cache}"
export LOGGING_LEVEL="${LOGGING_LEVEL:-INFO}"

# --- Edit for your experiment (underscore task = one segment per auxiliary locator) ---
TASK="${TASK:-nli_toxicity}"
SOURCE_DATA="${SOURCE_DATA:-${MUCOCO_ROOT}/new_module/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set.jsonl}"
REPORT_JSON="${REPORT_JSON:-${MUCOCO_ROOT}/new_module/_outputs/run_locate_union_masks_encoded_id_check_${SLURM_JOB_ID:-local}.json}"
MAX_PROMPTS="${MAX_PROMPTS:-}"

CACHE_DIR="${CACHE_DIR:-${HF_HOME}}"

LM_PATH="${LM_PATH:-gpt2-large}"
TOK_LM="${TOKENIZER_LM_PATH:-${LM_PATH}}"

# Auxiliary models (must match number of underscores in TASK plus one LM in --losses)
AUX1_PATH="${AUX1_PATH:-/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/}"
TOK_AUX1="${TOKENIZER_AUX1_PATH:-${AUX1_PATH}}"

AUX2_PATH="${AUX2_PATH:-/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint}"
TOK_AUX2="${TOKENIZER_AUX2_PATH:-${AUX2_PATH}}"

# argparse expects one argv per integer; never quote "${TARGET_IDS}" as a single string.
read -r -a TARGET_LABEL_IDS_ARRAY <<< "${TARGET_IDS:-1 1 0}"

LOSS_PY_ARGS=(
    --task "${TASK}"
    --source_data "${SOURCE_DATA}"
    --jsonl_primary_key "${JSONL_PRIMARY_KEY:-prompt}"
    --jsonl_secondary_key "${JSONL_SECONDARY_KEY:-text}"
    --losses gpt2_no_prefix classification_no_prefix_logprobloss classification_no_prefix_logprobloss
    --model_paths "${LM_PATH}" "${AUX1_PATH}" "${AUX2_PATH}"
    --tokenizer_paths "${TOK_LM}" "${TOK_AUX1}" "${TOK_AUX2}"
    --model_types AutoModelForCausalLM EncoderModel AutoModelForSequenceClassification
    --target_label_ids "${TARGET_LABEL_IDS_ARRAY[@]}"
    --device cuda
    --cache_dir "${CACHE_DIR}"
    --locate_method "${LOCATE_METHOD:-grad_norm}"
    --locate_unit "${LOCATE_UNIT:-word}"
    --num_edit_token_per_step "${NUM_EDIT_TOKEN_PER_STEP:-7}"
    --report_json "${REPORT_JSON}"
    --log_every "${LOG_EVERY:-10}"
    --max_failure_examples_saved "${MAX_FAILURE_EXAMPLES:-50}"
)
if [[ -n "${MAX_PROMPTS}" ]]; then
    LOSS_PY_ARGS+=(--max_prompts "${MAX_PROMPTS}")
fi
if [[ -n "${START_PROMPT:-}" ]]; then
    LOSS_PY_ARGS+=(--start_prompt "${START_PROMPT}")
fi

exec srun python -m new_module.run_locate_union_masks_encoded_id_check "${LOSS_PY_ARGS[@]}"
