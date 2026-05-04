#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=###
#SBATCH --output='new_module/_slurm_outs/####_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# RESULT_PATHS_CSV=/home/hyeryung/data/mucoco/new_module/dev_utils/input/nli_result_paths.csv
# # All paths from jsonl_path column (header skipped); empty lines ignored
# mapfile -t OUTPUT_FILES < <(tail -n +2 "$RESULT_PATHS_CSV" | sed '/^[[:space:]]*$/d')

# srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/dev_utils/calc_editedsampleonly_metrics_multiple_files.py \
#   --output_files "${OUTPUT_FILES[@]}" \
#   --index_files /home/hyeryung/data/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_edit_candidates_0_48.indexes_in_3105.txt \
#   --nicknames th0.48 \
#   --task nli \
#   --output_dir /home/hyeryung/data/mucoco/new_module/dev_utils/output \
#   --aggregate_csv nli_th0.48_editedonly_metrics.csv

RESULT_PATHS_CSV=/home/hyeryung/data/mucoco/new_module/dev_utils/input/toxicity_result_paths.csv
# All paths from jsonl_path column (header skipped); empty lines ignored
mapfile -t OUTPUT_FILES < <(tail -n +2 "$RESULT_PATHS_CSV" | sed '/^[[:space:]]*$/d')

srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/dev_utils/calc_editedsampleonly_metrics_multiple_files.py \
  --output_files "${OUTPUT_FILES[@]}" \
  --index_files /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_edit_candidates_0_39.indexes_in_0_95.txt \
  --nicknames th0.39 \
  --task toxicity \
  --output_dir /home/hyeryung/data/mucoco/new_module/dev_utils/output \
  --aggregate_csv toxicity_th0.39_editedonly_metrics.csv