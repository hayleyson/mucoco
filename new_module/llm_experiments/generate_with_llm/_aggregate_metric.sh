python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/_aggregate_metrics.py \
  --output-xlsx /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/_aggregated_metrics.xlsx \
  --models gpt-oss-20b Qwen3-8B Llama-3.1-8B-Instruct \
  --tasks nli_nontoxic \
  --prompt-suffixes plain consistent_0shot nontoxic_0shot consistent_nontoxic_0shot
