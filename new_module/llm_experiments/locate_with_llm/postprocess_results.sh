#!/bin/bash

# Process all files in the results directory
results_dir="new_module/llm_experiments/locate_with_llm/results"
original_toxic_file="new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl"
original_inconsistent_file="new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl"
output_dir="new_module/llm_experiments/locate_with_llm/processed_results"

python new_module/llm_experiments/locate_with_llm/postprocess_results.py \
--results_dir $results_dir \
--original_toxic_file $original_toxic_file \
--original_inconsistent_file $original_inconsistent_file \
--output_dir $output_dir