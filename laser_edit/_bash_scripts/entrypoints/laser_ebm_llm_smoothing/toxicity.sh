#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --job-name=refine_tox
#SBATCH --output='laser_edit/_slurm_outs/refine_tox_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: LASER & EBM edit + LLM smoothing | Task: toxicity
# Source: edit/ebm/llm_smoothing/run_refine.sh
# Run AFTER the corresponding laser_ebm/toxicity.sh job finishes.
# Update INPUT_PATH to the EBM run directory outputs.txt.

REPO_ROOT="/home/hyeryung/data/mucoco"
cd "$REPO_ROOT" || exit 1

FEW_SHOT_PATH="${REPO_ROOT}/laser_edit/edit/ebm/llm_smoothing/few_shot_prompts.json"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG="${MODEL_NAME##*/}"
TEMPLATE_VERSION="1"

ORIGINAL_TEXT_PATH="/home/hyeryung/data/mucoco/laser_edit/data/toxicity-avoidance/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
INPUT_PATH="/home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/ebm/2ao8yry0/outputs.txt"
BASE_DIR=$(dirname "$INPUT_PATH")
BASE_NAME=$(basename "$INPUT_PATH")
if [[ "$BASE_NAME" =~ \.txt\.[0-9]+$ ]]; then
    BASE_NAME="${BASE_NAME%.txt.*}"
elif [[ "$BASE_NAME" == *.txt ]]; then
    BASE_NAME="${BASE_NAME%.txt}"
fi

NUM_SHOTS=0
BATCH_SIZE=32
TASK="toxicity"
TOP_P=0.96
TIMESTAMP="$(date +%s)"
OUTPUT_PATH="${BASE_DIR}/${BASE_NAME}_${MODEL_TAG}_v${TEMPLATE_VERSION}_s${NUM_SHOTS}_p${TOP_P}_refined_${TIMESTAMP}.jsonl"

REFINE_ARGS=(
    --input_path "$INPUT_PATH"
    --output_path "$OUTPUT_PATH"
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --template_version "$TEMPLATE_VERSION"
    --num_shots "$NUM_SHOTS"
    --top_p "$TOP_P"
    --batch_size "$BATCH_SIZE"
    --original_text_path "$ORIGINAL_TEXT_PATH"
    --use_vllm
)
if (( NUM_SHOTS > 0 )); then
    REFINE_ARGS+=(--few_shot_path "$FEW_SHOT_PATH")
fi

srun python "${REPO_ROOT}/laser_edit/edit/ebm/llm_smoothing/refine.py" \
    "${REFINE_ARGS[@]}"
