#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --output='laser_edit/_slurm_outs/eval_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache


RUN_PATH=""
GENERATIONS_FILE_PATH="/home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/llm/final/toxicity_notmasked_loc_edit_200891.jsonl"
# METRICS="set-consistency,ppl-qwen,dist-n,repetition,fluency,contents-preservation"
# METRICS="toxicity,ppl-qwen,dist-n,repetition,contents-preservation"
# METRICS="nli,toxicity,nli_toxicity_joint,ppl-qwen,fluency,dist-n,repetition,contents-preservation"
METRICS="toxicity,ppl-qwen,dist-n,repetition,contents-preservation"
# SOURCE_FILE_PATH="/home/hyeryung/data/mucoco/laser_edit/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl"
# SOURCE_FILE_PATH="/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"
SOURCE_FILE_PATH="/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
TASK="toxicity"

srun -n 1 -c 1 python laser_edit/evaluation/run_evaluation.py \
  --run_path "${RUN_PATH}" \
  --generations_file_path "${GENERATIONS_FILE_PATH}" \
  --metrics "${METRICS}" \
  --source_file_path "${SOURCE_FILE_PATH}" \
  --task "${TASK}"
  # --set_consistency_llm_edit_output