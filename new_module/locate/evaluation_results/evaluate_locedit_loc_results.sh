#!/bin/bash

results_dir="new_module/locate/results"
output_dir="new_module/locate/evaluation_results"

python new_module/locate/evaluation_results/evaluate_locedit_loc_results.py \
--results_dir $results_dir \
--output_dir $output_dir