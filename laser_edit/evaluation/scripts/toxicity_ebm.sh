#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=eval_tox_ebm
#SBATCH --output='laser_edit/_slurm_outs/eval_toxicity_ebm_%j.out'

# Evaluate toxicity-avoidance LaSEr-EBM Edit outputs (no LLM postprocess).
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
METRICS="toxicity,ppl-qwen,dist-n,repetition,fluency,contents-preservation"
TASK="toxicity"
SOURCE_FILE_PATH="${ROOT}/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"

GENERATIONS_FILE_PATH="${ROOT}/outputs/toxicity/gpt3_5_gen/ebm/2ao8yry0/outputs.txt"

srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "${METRICS}" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"

echo "Done: ${GENERATIONS_FILE_PATH}-results.txt"
