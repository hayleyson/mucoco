#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=eval_multi
#SBATCH --output='laser_edit/_slurm_outs/eval_multi_%j.out'

# Evaluate joint NLI + toxicity (multi-constraint) edit outputs.
# Update GENERATIONS_FILE_PATH to the run you want to score.

set -euo pipefail

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

ROOT="/home/hyeryung/data/mucoco"
cd "$ROOT"

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

RUN_PATH=""
METRICS="nli,toxicity,nli_toxicity_joint,ppl-qwen,dist-n,repetition,contents-preservation"
TASK="nli_toxicity"
SOURCE_FILE_PATH="${ROOT}/laser_edit/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl"

# Examples:
#   LLM:  outputs/nli_toxicity/llm/<exp>_loc_edit_<job>.jsonl  (or .../llm/final/)
#   EBM:  outputs/nli_toxicity/ebm/<wandb_id>/outputs.txt
GENERATIONS_FILE_PATH="${ROOT}/outputs/nli_toxicity/ebm/f2hjzr5f/outputs.txt"

srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "${METRICS}" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"

echo "Done: ${GENERATIONS_FILE_PATH}-results.txt"
