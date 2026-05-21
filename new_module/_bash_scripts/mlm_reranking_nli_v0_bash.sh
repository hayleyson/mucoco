# python new_module/ebm_edit_main_debug.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 3 \
# --loss_weights 0.1 1.0 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task nli \
# --output_dir_prefix 'outputs/nli/' \
# --source_data '/home/hyeryung/data/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl' \
# --source_style 'inconsistent' \
# --target_style 'consistent' \
# --target_label_ids 1 1 \
# --thresholds 0.99 \
# --wandb_project 'nli-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --tokenizer_paths 'gpt2-large' '/data3/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/' \
# --locate_method 'grad_norm' \
# --losses gpt2_no_prefix classification \
# --model_types AutoModelForCausalLM EncoderModel

python new_module/ebm_edit_main_debug.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 10 \
--n_iter 3 \
--loss_weights 0.1 1.0 \
--selection_criteria allsat_primary \
--cache_dir '/home/hyeryung/data/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task toxicity \
--output_dir_prefix 'outputs/toxicity/llm' \
--source_data '/home/hyeryung/data/mucoco/new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--thresholds 0.998671 \
--wandb_project 'toxicity-decoding' \
--model_paths 'google/gemma-2-2b' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/' \
--tokenizer_paths 'google/gemma-2-2b' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/' \
--locate_method 'grad_norm' \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification