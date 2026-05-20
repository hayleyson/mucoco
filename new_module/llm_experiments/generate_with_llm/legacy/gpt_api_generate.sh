#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=gen_gpt4_toxic
#SBATCH --output='/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/logs/%j.out'

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# Set parameters
MODEL_NAME=gpt-5-mini-2025-08-07
MODEL_ACCESS_METHOD=api
PROMPT_TYPE=nli_plain
TASK=anli-r2-test
FILE_SAVE_DIR=new_module/llm_experiments/generate_with_llm/baselm_gens/${MODEL_NAME##*/}/nli
INPUT_FILE_PATH=new_module/data/logical-consistency/anli-r2-test_premises.jsonl
NUM_TEST_PROMPTS=1
NUM_RETURN_SEQUENCES=10
TOP_P=0.96
MAX_NEW_TOKENS=150


# Generate file save path
TIMESTAMP=$(date +%Y%m%d%H%M%S)
FILE_SAVE_PATH="${FILE_SAVE_DIR}/${MODEL_NAME##*/}_${TASK}_${PROMPT_TYPE}_${TIMESTAMP}.jsonl"

srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/gpt_api_generate.py \
--model $MODEL_NAME \
--openai_api_key $OPENAI_API_KEY \
--file_save_path $FILE_SAVE_PATH \
--input_file_path $INPUT_FILE_PATH \
--prompt_type $PROMPT_TYPE \
--max_tokens $MAX_NEW_TOKENS \
--num_test_prompts $NUM_TEST_PROMPTS \
--num_return_sequences $NUM_RETURN_SEQUENCES

