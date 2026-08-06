#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=sl_tox_pp
#SBATCH --output='laser_edit/_slurm_outs/self_loc_tox_pp_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: Self-locate & LLM edit | Task: toxicity | Step: postprocess
# Source: postprocess_results.sh (commented baselm_gens_nontoxic block)

result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_toxic_5shot_type1_v3_baselm_gens_nontoxic_1782821550.jsonl"
original_file="/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
output_dir="laser_edit/locate/llm/nli_toxicity/processed_results"
mkdir -p "$output_dir"

srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
--result_file "$result_file" \
--original_file "$original_file" \
--output_file "$output_dir/$(basename "${result_file%.jsonl}")_processed.jsonl" \
--task baselm_gens_nontoxic
