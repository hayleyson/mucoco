#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=locate_with_llm
#SBATCH --output='laser_edit/_slurm_outs/locate_with_llm_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# GPTs
# Thinking models
# o3-2025-04-16
# o4-mini-2025-04-16
# gpt-5-nano-2025-08-07
# gpt-5-2025-08-07
# gpt-5-mini-2025-08-07
# gpt-5.4-2026-03-05

# Non-thinking models
# gpt-4.1-mini-2025-04-14
# gpt-4.1-nano-2025-04-14
# gpt-4.1-2025-04-14
# gpt-5.4-2026-03-05


# MEMO
# I ran the following code for these models:
# gpt-4.1-2025-04-14
# gpt-4.1-nano-2025-04-14
# gpt-5-nano-2025-08-07
# gpt-5-2025-08-07

# SETTING COMMON VARIABLES
NUM_TEST_PROMPTS=-1
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
REASONING_EFFORT="medium"
MAX_TOKENS=5000
TOP_P=1e-10

REPETITIONS=1

# PROMPT_LISTS=("locate_toxic_incon_5shot_type1")
PROMPT_LISTS=("locate_toxic_5shot_type1_v3")

for i in $(seq 1 $REPETITIONS); do
    for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
        srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/locate_with_llm.py \
        $MODEL_NAME \
        $PROMPT_TYPE \
        "nli_toxicity_rewrite_hypothesis_toxic" \
        --num_test_prompts $NUM_TEST_PROMPTS \
        --max_tokens $MAX_TOKENS \
        --top_p $TOP_P \
        --temperature 0.0 \
        --reasoning_effort $REASONING_EFFORT \
        --use_vllm
    done
done

PROMPT_LISTS=("locate_incon_5shot_type1_v3")

for i in $(seq 1 $REPETITIONS); do
    for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
        srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/locate_with_llm.py \
        $MODEL_NAME \
        $PROMPT_TYPE \
        "nli_toxicity_rewrite_hypothesis_toxic" \
        --num_test_prompts $NUM_TEST_PROMPTS \
        --max_tokens $MAX_TOKENS \
        --top_p $TOP_P \
        --temperature 0.0 \
        --reasoning_effort $REASONING_EFFORT \
        --use_vllm
    done
done

exit 0



# RUN BASE LM GENERATIONS: GPT-3.5 nontoxic RealToxicityPrompts generations
PROMPT_LISTS=("locate_toxic_5shot_type1_v3")

for i in $(seq 1 $REPETITIONS); do
    for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
        srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/locate_with_llm.py \
        $MODEL_NAME \
        $PROMPT_TYPE \
        "baselm_gens_nontoxic" \
        --num_test_prompts $NUM_TEST_PROMPTS \
        --max_tokens $MAX_TOKENS \
        --top_p $TOP_P \
        --temperature 0.0 \
        --reasoning_effort $REASONING_EFFORT
    done
done

# RUN BASE LM GENERATIONS: ANLI-R2 consistent logical-consistency generations
PROMPT_LISTS=("locate_incon_5shot_type1_v3")

for i in $(seq 1 $REPETITIONS); do
    for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
        srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/locate_with_llm.py \
        $MODEL_NAME \
        $PROMPT_TYPE \
        "baselm_gens_consistent" \
        --num_test_prompts $NUM_TEST_PROMPTS \
        --max_tokens $MAX_TOKENS \
        --top_p $TOP_P \
        --temperature 0.0 \
        --reasoning_effort $REASONING_EFFORT
    done
done

exit 0


# # RUN TOXIC SPANS EXTENDED
# # PROMPT_LISTS=("locate_toxic_0shot_type1_v3_text_version" "locate_toxic_5shot_type1_v3_text_version" "locate_toxic_5shot_cot_type1_v4_text_version")
# PROMPT_LISTS=("locate_toxic_0shot_type1_v3_text_version" "locate_toxic_5shot_type1_v3_text_version")

# for i in $(seq 1 $REPETITIONS); do
#     for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
#         srun python laser_edit/llm_experiments/locate_with_llm/locate_with_llm.py \
#         $MODEL_NAME \
#         $PROMPT_TYPE \
#         "toxicspans_extended" \
#         --num_test_prompts $NUM_TEST_PROMPTS \
#         --max_tokens $MAX_TOKENS \
#         --top_p $TOP_P \
#         --temperature 0.0
#     done
# done


# # RUN TOXIC SPANS
# # PROMPT_LISTS=("locate_toxic_0shot_type1_v3" "locate_toxic_5shot_type1_v3" "locate_toxic_5shot_cot_type1_v4")
PROMPT_LISTS=("locate_toxic_5shot_type1_v3")

for i in $(seq 1 $REPETITIONS); do
    for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
        srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/locate_with_llm.py \
        $MODEL_NAME \
        $PROMPT_TYPE \
        "toxicspans" \
        --num_test_prompts $NUM_TEST_PROMPTS \
        --max_tokens $MAX_TOKENS \
        --top_p $TOP_P \
        --temperature 0.0 \
        --reasoning_effort $REASONING_EFFORT
    done
done

# # RUN INCONSISTENT SPANS
# # PROMPT_LISTS=("locate_incon_0shot_type1_v3" "locate_incon_5shot_type1_v3" "locate_incon_5shot_cot_type1_v4")
PROMPT_LISTS=("locate_incon_5shot_type1_v3")

for i in $(seq 1 $REPETITIONS); do
    for PROMPT_TYPE in "${PROMPT_LISTS[@]}"; do
        srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/locate_with_llm.py \
        $MODEL_NAME \
        $PROMPT_TYPE \
        "inconsistentspans" \
        --num_test_prompts $NUM_TEST_PROMPTS \
        --max_tokens $MAX_TOKENS \
        --top_p $TOP_P \
        --temperature 0.0 \
        --reasoning_effort $REASONING_EFFORT
    done
done

# # RUN BIG-BENCH-MISTAKE

# PROMPT_TYPE="direct"
# DATASET="logical_deduction"
# NUM_TEST_PROMPTS=-1

# srun -n 1 -c 1 python laser_edit/llm_experiments/locate_with_llm/locate_with_llm.py \
# $MODEL_NAME \
# $PROMPT_TYPE \
# $DATASET \
# --num_test_prompts $NUM_TEST_PROMPTS \
# --max_tokens $MAX_TOKENS \
# --top_p $TOP_P \
# --temperature 0.0

# PROMPT_TYPE="direct"
# DATASET="tracking_shuffled_objects"
# NUM_TEST_PROMPTS=-1

# srun -n 1 -c 1 python laser_edit/llm_experiments/locate_with_llm/locate_with_llm.py \
# $MODEL_NAME \
# $PROMPT_TYPE \
# $DATASET \
# --num_test_prompts $NUM_TEST_PROMPTS \
# --max_tokens $MAX_TOKENS \
# --top_p $TOP_P \
# --temperature 0.0