#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --output='laser_edit/_slurm_outs/eval_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache


RUN_PATH=""
GENERATIONS_FILE_PATH="outputs/sc_energy/set_nli/ebm/0cyf2b8p/outputs.txt"
# METRICS="set-consistency,ppl-qwen,dist-n,repetition,fluency,contents-preservation"
METRICS="set-consistency-gpt-5.4-mini"
SOURCE_FILE_PATH="laser_edit/data/set_nli/testset_incon_300/set_nli_testset_incon_300.jsonl"
TASK="set_nli"

srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "${METRICS}" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"
  # --set_consistency_llm_edit_output
