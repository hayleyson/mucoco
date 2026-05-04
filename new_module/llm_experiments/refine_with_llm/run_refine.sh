#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=refine_test
#SBATCH --output='new_module/_slurm_outs/refine_test_%A_%a.out'
#SBATCH --array=0

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export HF_DATASETS_CACHE=/home/hyeryung/data/hf_cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/hf_cache

# Define input paths for each array task
INPUT_PATHS=(
    /home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/ebm/j18pi8ab/outputs_epsilon0.95.txt.0
    # "outputs/sc_energy/set_lconvqa/6vzkroqi/outputs.txt"
    # "outputs/sc_energy/set_lconvqa/b5y3ql1w/outputs.txt"
    # "outputs/sc_energy/set_lconvqa/9dffjesl/outputs.txt"
    # "outputs/sc_energy/set_lconvqa/txix56dv/outputs.txt"
)

INPUT_PATH=${INPUT_PATHS[$SLURM_ARRAY_TASK_ID]}
# Derive output directory and base name
BASE_DIR=$(dirname "$INPUT_PATH")
BASE_NAME=$(basename "$INPUT_PATH" .txt)

NUM_SHOTS=0
BATCH_SIZE=1
FEW_SHOT_PATH="new_module/llm_experiments/refine_with_llm/few_shot_prompts_refine.json"
TASK="toxicity"

# Run 1: top_p = 0.96
TOP_P=0.96
OUTPUT_PATH="${BASE_DIR}/${BASE_NAME}_qwen2.5_7B_s${NUM_SHOTS}_p${TOP_P}_refined_concise_prompt.jsonl"

srun python /home/hyeryung/data/mucoco/new_module/llm_experiments/refine_with_llm/refine.py \
--input_path "$INPUT_PATH" \
--output_path "$OUTPUT_PATH" \
--num_shots "$NUM_SHOTS" \
--top_p "$TOP_P" \
--batch_size "$BATCH_SIZE" \
--few_shot_path "$FEW_SHOT_PATH" \
--task "$TASK" \
--consider_prefix
