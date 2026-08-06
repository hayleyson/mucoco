#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=eval_tox_llm
#SBATCH --output='laser_edit/_slurm_outs/eval_toxicity_llm_%j.out'

# Evaluate toxicity-avoidance LLM-edit outputs.
# Pipeline:
#   1) evaluate toxicity + other non-fluency metrics on the original generations file
#   2) postprocess for_fluency_metric -> evaluate fluency
# Update GENERATIONS_FILE_PATH to the LLM-edit JSONL you want to score.

set -euo pipefail

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

ROOT="/home/hyeryung/data/mucoco"
cd "$ROOT"

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

RUN_PATH=""
TASK="toxicity"
SOURCE_FILE_PATH="${ROOT}/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"

GENERATIONS_FILE_PATH="${ROOT}/outputs/toxicity/gpt3_5_gen/llm/final/nontoxic_masked_self_locate_loc_edit_200817.jsonl"

BASE_NAME="$(basename "${GENERATIONS_FILE_PATH}")"
GEN_DIR="$(dirname "${GENERATIONS_FILE_PATH}")"
if [[ "$(basename "${GEN_DIR}")" == "final" ]]; then
  LLM_ROOT="$(dirname "${GEN_DIR}")"
  FLUENCY_PP_DIR="${LLM_ROOT}/fluency"
else
  FLUENCY_PP_DIR="${GEN_DIR}/fluency_postprocessed"
fi
mkdir -p "${FLUENCY_PP_DIR}"

echo "=== Evaluate toxicity/ppl/dist/repetition/contents-preservation on original ${GENERATIONS_FILE_PATH} ==="
srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "toxicity,ppl-qwen,dist-n,repetition,contents-preservation" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT
cp "${GENERATIONS_FILE_PATH}" "${STAGE_DIR}/${BASE_NAME}"

echo "=== Postprocess for fluency metric ==="
srun -n 1 -c 1 python laser_edit/evaluation/postprocess_generations.py \
  --input_dir "${STAGE_DIR}" \
  --save_dir "${FLUENCY_PP_DIR}" \
  --suffix jsonl \
  --option for_fluency_metric

FLUENCY_PP_FILE="${FLUENCY_PP_DIR}/${BASE_NAME}"
echo "=== Evaluate fluency on ${FLUENCY_PP_FILE} ==="
srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${FLUENCY_PP_FILE}" \
  --metrics "fluency" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"

echo "Done."
echo "  Other metrics:   ${GENERATIONS_FILE_PATH}-results.txt"
echo "  Fluency results: ${FLUENCY_PP_FILE}-results.txt"
