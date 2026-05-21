#!/bin/bash

results_dir="new_module/llm_experiments/locate_with_llm/processed_results"
output_dir="new_module/llm_experiments/locate_with_llm/evaluation_results"

python new_module/llm_experiments/locate_with_llm/evaluate_llm_results.py \
--results_dir $results_dir \
--output_dir $output_dir