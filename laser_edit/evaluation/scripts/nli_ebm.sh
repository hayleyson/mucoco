#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=eval_nli_ebm
#SBATCH --output='laser_edit/_slurm_outs/eval_nli_ebm_%j.out'

# Evaluate contradiction-avoidance (NLI) LaSEr-EBM Edit outputs (no LLM postprocess).
# Update GENERATIONS_FILE_PATH to the EBM run outputs.txt you want to score.

set -euo pipefail

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

ROOT="/home/hyeryung/data/mucoco"
cd "$ROOT"

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

RUN_PATH=""
METRICS="nli,ppl-qwen,dist-n,repetition,fluency,contents-preservation"
TASK="nli"
SOURCE_FILE_PATH="${ROOT}/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"

GENERATIONS_FILE_PATH="${ROOT}/outputs/nli/ebm/31tdw0g4/outputs.txt"

srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "${METRICS}" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"

echo "Done: ${GENERATIONS_FILE_PATH}-results.txt"
