#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --job-name=refine_test
#SBATCH --output='laser_edit/_slurm_outs/refine_test_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

REPO_ROOT="/home/hyeryung/data/mucoco"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

# Non-changing values
FEW_SHOT_PATH="${REPO_ROOT}/laser_edit/edit/ebm/llm_smoothing/few_shot_prompts.json"

# HuggingFace model id (must match refine.py --model_name)
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# Short name for output files: org/model -> model (after last /)
MODEL_TAG="${MODEL_NAME##*/}"

# grammar_refinement_versions key in prompt_templates.json (e.g. 1, 2)
TEMPLATE_VERSION="1"


ORIGINAL_TEXT_PATH="${REPO_ROOT}/laser_edit/data/set_nli/testset_incon_300/set_nli_testset_incon_300.jsonl"

INPUT_PATH="${REPO_ROOT}/outputs/sc_energy/set_nli/ebm/0cyf2b8p/outputs.txt"
BASE_DIR=$(dirname "$INPUT_PATH")
BASE_NAME=$(basename "$INPUT_PATH")
# Strip .txt.N (shard / part index) or plain .txt for output naming
if [[ "$BASE_NAME" =~ \.txt\.[0-9]+$ ]]; then
    BASE_NAME="${BASE_NAME%.txt.*}"
elif [[ "$BASE_NAME" == *.txt ]]; then
    BASE_NAME="${BASE_NAME%.txt}"
fi

NUM_SHOTS=0
BATCH_SIZE=32
TASK="set_nli"

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
