#!/bin/bash

results_dir="laser_edit/locate/results"
output_dir="laser_edit/locate/evaluation_results"

python laser_edit/locate/evaluation_results/evaluate_locedit_loc_results.py \
--results_dir $results_dir \
--output_dir $output_dir