#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=master
#SBATCH --job-name=gen_llm
#SBATCH --output='/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/logs/%j.out'

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# Supported models
# claude-sonnet-4-6
# gemini-2.5-flash
# gpt-5-mini-2025-08-07
# gpt-3.5-turbo-0120
# Qwen/Qwen3-8B
# openai/gpt-oss-20b
# meta-llama/Llama-3.1-8B-Instruct

# NOTE. For Qwen/Qwen3-8B, set top_p=0.95, top_k=20, temperature=0.6
# For Others, set top_p=0.96, top_k=50, temperature=1.0

# Set parameters
MODEL_NAME=openai/gpt-oss-20b
MODEL_ACCESS_METHOD=vllm
PROMPT_TYPE=nli+nontoxic_plain
NUM_SHOTS=5
TASK=nli_nontoxic # nli: anli-r2-test / toxicity: nontoxic
FILE_SAVE_DIR=new_module/llm_experiments/generate_with_llm/baselm_gens/${MODEL_NAME##*/}/${TASK}
# nli: /home/hyeryung/data/mucoco/new_module/data/logical-consistency/anli-r2-test_premises.jsonl
# nontoxic: /home/hyeryung/data/mucoco/new_module/data/toxicity-avoidance/nontoxic_prompts-250.jsonl
# nli+nontoxic: /home/hyeryung/data/mucoco/new_module/data/nli-toxicity/nontoxic-prompts-toxic-continuations.jsonl
INPUT_FILE_PATH=/home/hyeryung/data/mucoco/new_module/data/nli-toxicity/nontoxic-prompts-toxic-continuations.jsonl
NUM_TEST_PROMPTS=-1
NUM_RETURN_SEQUENCES=10
TOP_P=0.96
TOP_K=50
TEMPERATURE=1.0
MAX_NEW_TOKENS=4096

# Generate file save path
TIMESTAMP=$(date +%Y%m%d%H%M%S)
if [[ "$PROMPT_TYPE" == *few_shot* ]]; then
  FILE_SAVE_PATH="${FILE_SAVE_DIR}/${MODEL_NAME##*/}_${TASK}_${PROMPT_TYPE}_${NUM_SHOTS}shot_${TIMESTAMP}.jsonl"
else
  FILE_SAVE_PATH="${FILE_SAVE_DIR}/${MODEL_NAME##*/}_${TASK}_${PROMPT_TYPE}_${TIMESTAMP}.jsonl"
fi

echo "MODEL_NAME: ${MODEL_NAME}"
echo "PROMPT_TYPE: ${PROMPT_TYPE}"
echo "NUM_SHOTS: ${NUM_SHOTS}"

srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/generate_main.py \
--model $MODEL_NAME \
--file_save_path $FILE_SAVE_PATH \
--input_file_path $INPUT_FILE_PATH \
--prompt_type $PROMPT_TYPE \
--num_shots $NUM_SHOTS \
--max_new_tokens $MAX_NEW_TOKENS \
--num_test_prompts $NUM_TEST_PROMPTS \
--num_return_sequences $NUM_RETURN_SEQUENCES \
--model_access_method $MODEL_ACCESS_METHOD \
--top_p $TOP_P \
--top_k $TOP_K \
--temperature $TEMPERATURE
