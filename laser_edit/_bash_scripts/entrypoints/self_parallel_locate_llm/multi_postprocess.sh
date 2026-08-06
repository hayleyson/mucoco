#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=spl_multi_pp
#SBATCH --output='laser_edit/_slurm_outs/self_par_multi_pp_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: Self-parallel-locate & LLM edit | Task: multi | Step: postprocess (union)
# Source: postprocess_results.sh active block (--result_file + --result_file_2)

consistent_result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_incon_5shot_type1_v3_nli_toxicity_rewrite_hypothesis_toxic_1783044705.jsonl"
nontoxic_result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_toxic_5shot_type1_v3_nli_toxicity_rewrite_hypothesis_toxic_1783044508.jsonl"
original_file="/home/hyeryung/data/mucoco/laser_edit/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl"
output_dir="laser_edit/locate/llm/nli_toxicity/processed_results"
mkdir -p "$output_dir"

# Update filenames below to match the locate job IDs from multi_locate.sh
srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
--result_file "$consistent_result_file" \
--result_file_2 "$nontoxic_result_file" \
--original_file "$original_file" \
--output_file "$output_dir/Qwen2.5-7B-Instruct_locate_incon_5shot_type1_v3_toxic_5shot_type1_v3_union_nli_toxicity_rewrite_hypothesis_toxic_1783044705_1783044508_processed.jsonl" \
--task nli_toxicity_rewrite_hypothesis_toxic
