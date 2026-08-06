#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=sl_nli_loc
#SBATCH --output='laser_edit/_slurm_outs/self_loc_nli_locate_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: Self-locate & LLM edit | Task: nli | Step: locate
# Source: locate_with_llm.sh (commented baselm_gens_consistent block)

NUM_TEST_PROMPTS=-1
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
REASONING_EFFORT="medium"
MAX_TOKENS=5000
TOP_P=1e-10
PROMPT_LISTS=("locate_incon_5shot_type1_v3")

for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
    srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/locate_with_llm.py \
    $MODEL_NAME \
    $PROMPT_TYPE \
    "baselm_gens_consistent" \
    --num_test_prompts $NUM_TEST_PROMPTS \
    --max_tokens $MAX_TOKENS \
    --top_p $TOP_P \
    --temperature 0.0 \
    --reasoning_effort $REASONING_EFFORT \
    --use_vllm
done
