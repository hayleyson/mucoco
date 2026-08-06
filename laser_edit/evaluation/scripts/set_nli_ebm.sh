#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=eval_setnli_ebm
#SBATCH --output='laser_edit/_slurm_outs/eval_set_nli_ebm_%j.out'

# Evaluate set-NLI (Set-SNLI) LaSEr-EBM Edit outputs.txt (no --set_consistency_llm_edit_output).
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
TASK="set_nli"
SOURCE_FILE_PATH="${ROOT}/laser_edit/data/set_nli/testset_incon_300/set_nli_testset_incon_300.jsonl"

GENERATIONS_FILE_PATH="${ROOT}/outputs/sc_energy/set_nli/ebm/<wandb_id>/outputs.txt"

srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "${METRICS}" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"

echo "Done: ${GENERATIONS_FILE_PATH}-results.txt"
