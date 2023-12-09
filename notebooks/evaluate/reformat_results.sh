# python notebooks/evaluate/reformat_results.py \
# --input_path /home/s3/hyeryung/mucoco/outputs/toxicity/locate-unit/gpt2-word-netps-1-nls-1-nps1-modelmucola/outputs_epsilon-3.txt \
# --output_path /home/s3/hyeryung/mucoco/notebooks/evaluate/xlsx_outputs/mucola_result.xlsx \
# --nickname mucola

# python notebooks/evaluate/reformat_results.py \
# --input_path /home/s3/hyeryung/mucoco/new_module/toxicity-avoidance/data/testset_gpt2_2500.jsonl \
# --output_path /home/s3/hyeryung/mucoco/notebooks/evaluate/xlsx_outputs/gpt2_result.xlsx \
# --nickname gpt2-original

## mlm-beamsearch-v1 result with lowest avg-max-toxicity energy score
# python notebooks/evaluate/reformat_results.py \
# --input_path /home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/sweep-mlm-beamsearch-v1-token-nps5-k10-beam3-allsat_primary-t7w3q9xu-wandb/outputs_epsilon-3.txt \
# --output_path /home/s3/hyeryung/mucoco/notebooks/evaluate/xlsx_outputs/mlm-beamsearch-v1-token-nps5-k10-beam3-allsat_primary-closs0.43-t7w3q9xu_result.xlsx \
# --nickname mlm-beamsearch-v1-token-nps5-k10-beam3-allsat_primary-closs0.43-t7w3q9xu

## locate-edit using mucola model result with lowest avg-max-toxicity energy score
# python notebooks/evaluate/reformat_results.py \
# --input_path /home/s3/hyeryung/mucoco/outputs/toxicity/locate-unit/gpt2-token-netps3-nls1-nps1-modelmucola/outputs_epsilon-3.txt \
# --output_path /home/s3/hyeryung/mucoco/notebooks/evaluate/xlsx_outputs/locate-unit-gpt2-token-netps3-nls1-nps1-modelmucola_result.xlsx \
# --nickname locate-unit-gpt2-token-netps3-nls1-nps1-modelmucola

## locate-edit using energy model result with lowest avg-max-toxicity energy score
# python notebooks/evaluate/reformat_results.py \
# --input_path /home/s3/hyeryung/mucoco/outputs/toxicity/locate-unit/gpt2-word-netps3-nls4-nps1-modelenergy/outputs_epsilon-3.txt \
# --output_path /home/s3/hyeryung/mucoco/notebooks/evaluate/xlsx_outputs/locate-unit-gpt2-word-netps3-nls4-nps1-modelenergy_result.xlsx \
# --nickname locate-unit-gpt2-word-netps3-nls4-nps1-modelenergy