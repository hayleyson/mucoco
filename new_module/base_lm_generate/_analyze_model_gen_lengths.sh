python /home/hyeryung/data/mucoco/new_module/base_lm_generate/_analyze_model_gen_lengths.py \
  --output-csv /home/hyeryung/data/mucoco/new_module/base_lm_generate/_analyzed_model_gen_lengths.csv \
  --models gpt-oss-20b Qwen3-8B Llama-3.1-8B-Instruct \
  --tasks com \
  --prompt-suffixes plain consistent_0shot nontoxic_0shot consistent_nontoxic_0shot
