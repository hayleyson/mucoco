#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=n04
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=postprocess_results      
#SBATCH --output='laser_edit/_slurm_outs/postprocess_results_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Original batch processing over the full results directory:
# results_dir="laser_edit/locate/llm/nli_toxicity/results"
# original_toxic_file="laser_edit/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl"
# original_inconsistent_file="laser_edit/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl"
# output_dir="laser_edit/locate/llm/nli_toxicity/processed_results"
#
# python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
# --results_dir $results_dir \
# --original_toxic_file $original_toxic_file \
# --original_inconsistent_file $original_inconsistent_file \
# --output_dir $output_dir

# # Process only the selected result files
# consistent_result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_incon_5shot_type1_v3_baselm_gens_consistent_1782821909.jsonl"
# nontoxic_result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_toxic_5shot_type1_v3_baselm_gens_nontoxic_1782821550.jsonl"
# original_baselm_gens_consistent_file="/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl"
# original_baselm_gens_nontoxic_file="/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
# output_dir="laser_edit/locate/llm/nli_toxicity/processed_results"

# mkdir -p "$output_dir"

# srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
# --result_file "$consistent_result_file" \
# --original_file "$original_baselm_gens_consistent_file" \
# --output_file "$output_dir/$(basename "${consistent_result_file%.jsonl}")_processed.jsonl" \
# --task baselm_gens_consistent

# srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
# --result_file "$nontoxic_result_file" \
# --original_file "$original_baselm_gens_nontoxic_file" \
# --output_file "$output_dir/$(basename "${nontoxic_result_file%.jsonl}")_processed.jsonl" \
# --task baselm_gens_nontoxic

# Process only the selected result files
consistent_result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_incon_5shot_type1_v3_nli_toxicity_rewrite_hypothesis_toxic_1783044705.jsonl"
nontoxic_result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_toxic_5shot_type1_v3_nli_toxicity_rewrite_hypothesis_toxic_1783044508.jsonl"
original_file="/home/hyeryung/data/mucoco/laser_edit/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl"
output_dir="laser_edit/locate/llm/nli_toxicity/processed_results"

mkdir -p "$output_dir"

srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
--result_file "$consistent_result_file" \
--result_file_2 "$nontoxic_result_file" \
--original_file "$original_file" \
--output_file "$output_dir/Qwen2.5-7B-Instruct_locate_incon_5shot_type1_v3_toxic_5shot_type1_v3_union_nli_toxicity_rewrite_hypothesis_toxic_1783044705_1783044508_processed.jsonl" \
--task nli_toxicity_rewrite_hypothesis_toxic
