RESULT_PATHS_CSV=/home/hyeryung/data/mucoco/laser_edit/utils/input/toxicity_result_paths.csv
# All paths from jsonl_path column (header skipped); empty lines ignored
mapfile -t OUTPUT_FILES < <(tail -n +2 "$RESULT_PATHS_CSV" | sed '/^[[:space:]]*$/d')

python /home/hyeryung/data/mucoco/laser_edit/utils/calc_editedsampleonly_metrics_multiple_files.py \
  --output_files "${OUTPUT_FILES[@]}" \
  --index_files /home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_edit_candidates_0_39.indexes_in_0_95.txt \
  --nicknames th0.39 \
  --task toxicity \
  --output_dir /home/hyeryung/data/mucoco/laser_edit/utils/output \
  --aggregate_csv toxicity_th0.39_editedonly_metrics.csv