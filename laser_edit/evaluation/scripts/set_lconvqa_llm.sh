#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=eval_lconvqa_llm
#SBATCH --output='laser_edit/_slurm_outs/eval_set_lconvqa_llm_%j.out'

# Evaluate set-LConVQA LLM-edit outputs (Plain / Self-locate / LaSEr-LLM JSONL).
# Always passes --set_consistency_llm_edit_output.
# Update GENERATIONS_FILE_PATH to the run you want to score.

set -euo pipefail

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

ROOT="/home/hyeryung/data/mucoco"
cd "$ROOT"

# Load API keys for set-consistency-gpt (optional if you drop that metric)
set -a
# shellcheck disable=SC1091
source "${ROOT}/.env"
set +a

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

RUN_PATH=""
METRICS="set-consistency,set-consistency-gpt,ppl-qwen,dist-n,repetition,fluency,contents-preservation"
TASK="set_lconvqa"
SOURCE_FILE_PATH="${ROOT}/laser_edit/data/lconvqa/testset_incon_300/lconvqa_testset_incon_300.jsonl"

GENERATIONS_FILE_PATH="${ROOT}/outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen2.5-7b-instruct_w_ebm_locate_edit_result.jsonl"

srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "${METRICS}" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}" \
  --set_consistency_llm_edit_output

echo "Done: ${GENERATIONS_FILE_PATH}-results.txt"
