#!/bin/bash
python new_module/locate/locate.py --method grad_norm \
--output_path "new_module/data/toxicity-avoidance/testset_gpt2_2500_locate_grad.jsonl" \
--input_path "new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
--model_name_or_path /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint \
--model_type RobertaCustomForSequenceClassification \
--batch_size 8