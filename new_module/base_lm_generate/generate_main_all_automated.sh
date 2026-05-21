#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --job-name=gen_llm
#SBATCH --array=0-3%1
#SBATCH --output='/home/hyeryung/data/mucoco/new_module/base_lm_generate/logs/%A_%a.out'

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
ROOT_DIR=/home/hyeryung/data/mucoco
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# Supported models
# anthropic/claude-sonnet-4-6
# gemini/gemini-2.5-flash
# openai/gpt-5-mini-2025-08-07
# Qwen/Qwen3-8B
# openai/gpt-oss-20b
# meta-llama/Llama-3.1-8B-Instruct

# --- User: set model only (HuggingFace id for vLLM, or API-style name for litellm) ---
MODEL_NAME=openai/gpt-oss-20b

# --- User: generation count controls ---
NUM_TEST_PROMPTS=-1
NUM_RETURN_SEQUENCES=1
PROMPT_START_INDEX=0

# --- Shared defaults ---
MAX_NEW_TOKENS=4096
GEN_PY="${ROOT_DIR}/new_module/base_lm_generate/generate_main.py"

# Prompt types run as one array task each (order must match #SBATCH --array max index).
PROMPT_TYPES=(
  # nli_plain
  # nli_0shot
  # nli_few_shot
  # nontoxic_plain
  # nontoxic_0shot
  # nontoxic_few_shot
  nli_nontoxic_plain
  nli_nontoxic_consistent_0shot
  nli_nontoxic_nontoxic_0shot
  nli_nontoxic_consistent_nontoxic_0shot
  # nli_nontoxic_few_shot
  # comment_plain
  # comment_consistent_0shot
  # comment_nontoxic_0shot
  # comment_consistent_nontoxic_0shot
)

# Infer API vs local backend from model name (same for all array tasks).
infer_model_access_method() {
  case "$MODEL_NAME" in
    anthropic/claude*|gemini/gemini*|openai/gpt-3.5*|openai/gpt-4*|openai/gpt-5*)
      echo litellm
      ;;
    *)
      echo vllm
      ;;
  esac
}

# Sampling defaults from repo notes: Qwen3 vs everything else.
infer_sampling() {
  if [[ "$MODEL_NAME" == *Qwen3* ]] || [[ "$MODEL_NAME" == Qwen/Qwen3-* ]]; then
    echo "0.95 20 0.6"
  else
    echo "0.96 50 1.0"
  fi
}

configure_prompt_type() {
  local pt="$1"
  PROMPT_TYPE="$pt"
  case "$pt" in
    nli_nontoxic_*|nli+nontoxic_*)
      TASK=nli_nontoxic
      INPUT_FILE_PATH="${ROOT_DIR}/new_module/data/nli-toxicity/nontoxic-prompts-toxic-continuations.jsonl"
      ;;
    nli_*)
      TASK=nli
      INPUT_FILE_PATH="${ROOT_DIR}/new_module/data/logical-consistency/anli-r2-test_premises.jsonl"
      ;;
    nontoxic_*)
      TASK=nontoxic
      INPUT_FILE_PATH="${ROOT_DIR}/new_module/data/toxicity-avoidance/nontoxic_prompts-250.jsonl"
      ;;
    comment_*)
      TASK=comment
      INPUT_FILE_PATH="${ROOT_DIR}/new_module/data/nli-toxicity/socc_gnm_top250_ncomments_10sampled_root_comments.jsonl"
      ;;
    senti_*|formal_*|informal_*)
      echo "configure_prompt_type: no default INPUT_FILE_PATH for '${pt}' — extend configure_prompt_type() in the script." >&2
      exit 1
      ;;
    *)
      echo "configure_prompt_type: unknown prompt type '${pt}'" >&2
      exit 1
      ;;
  esac

  if [[ "$pt" == *few_shot* ]]; then
    NUM_SHOTS=5
  else
    NUM_SHOTS=0
  fi
}

N_TYPES=${#PROMPT_TYPES[@]}
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_TYPES )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for PROMPT_TYPES (count=${N_TYPES}). Update #SBATCH --array=0-$((N_TYPES - 1))%3" >&2
  exit 1
fi

configure_prompt_type "${PROMPT_TYPES[$SLURM_ARRAY_TASK_ID]}"

# Map legacy names with + to what prompts.py expects.
case "$PROMPT_TYPE" in
  nli+nontoxic_plain) PROMPT_TYPE=nli_nontoxic_plain ;;
  nli+nontoxic_0shot) PROMPT_TYPE=nli_nontoxic_0shot ;;
  nli+nontoxic_few_shot) PROMPT_TYPE=nli_nontoxic_few_shot ;;
esac

MODEL_ACCESS_METHOD=$(infer_model_access_method)
read -r TOP_P TOP_K TEMPERATURE <<<"$(infer_sampling)"

FILE_SAVE_DIR="${ROOT_DIR}/new_module/base_lm_generate/baselm_gens/${MODEL_NAME##*/}/${TASK}"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
SAFE_PT="${PROMPT_TYPE//+/_}"
if [[ "$PROMPT_TYPE" == *few_shot* ]]; then
  FILE_SAVE_PATH="${FILE_SAVE_DIR}/${MODEL_NAME##*/}_${SAFE_PT}_${NUM_SHOTS}shot_${TIMESTAMP}_a${SLURM_ARRAY_TASK_ID}.jsonl"
else
  FILE_SAVE_PATH="${FILE_SAVE_DIR}/${MODEL_NAME##*/}_${SAFE_PT}_${TIMESTAMP}_a${SLURM_ARRAY_TASK_ID}.jsonl"
fi

echo "MODEL_NAME: ${MODEL_NAME}"
echo "MODEL_ACCESS_METHOD: ${MODEL_ACCESS_METHOD}"
echo "PROMPT_TYPE: ${PROMPT_TYPE}"
echo "TASK: ${TASK}"
echo "INPUT_FILE_PATH: ${INPUT_FILE_PATH}"
echo "NUM_SHOTS: ${NUM_SHOTS}"
echo "TOP_P/TOP_K/TEMPERATURE: ${TOP_P} ${TOP_K} ${TEMPERATURE}"
echo "ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-} TASK_ID: ${SLURM_ARRAY_TASK_ID}"

srun -n 1 -c 1 python "$GEN_PY" \
  --model "$MODEL_NAME" \
  --file_save_path "$FILE_SAVE_PATH" \
  --input_file_path "$INPUT_FILE_PATH" \
  --prompt_type "$PROMPT_TYPE" \
  --num_shots "$NUM_SHOTS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_test_prompts "$NUM_TEST_PROMPTS" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --model_access_method "$MODEL_ACCESS_METHOD" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --temperature "$TEMPERATURE" \
  --prompt_start_index "$PROMPT_START_INDEX"
