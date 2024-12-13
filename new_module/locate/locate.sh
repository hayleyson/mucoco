#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/loc_skiml_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

# python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "/data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint" \
# --input_file "new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl" \
# --output_file "new_module/locate/locate_num_tokens_eda/dev_set_negative_locate_max_5.jsonl" \
# --task sentiment \
# --label_id 0

# python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "/data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint" \
# --input_file "new_module/data/sentiment/dev_set_below_positive_threshold_778.jsonl" \
# --output_file "new_module/locate/locate_num_tokens_eda/dev_set_positive_locate_max_5.jsonl" \
# --task sentiment \
# --label_id 1

# python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
# --input_file "new_module/data/toxicity-avoidance/testset_gpt2_2500_below_nontoxic_threshold_468.jsonl" \
# --output_file "new_module/locate/locate_num_tokens_eda/testset_gpt2_2500_nontoxic_locate_max_5.jsonl" \
# --task toxicity \
# --label_id 0

# python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/" \
# --input_file "new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl" \
# --output_file "new_module/locate/locate_num_tokens_eda/anli-r2-test_consistent_locate_max_5.jsonl" \
# --task nli \
# --label_id 1

# python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "/data/hyeryung/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/" \
# --input_file "data/formality/GYAFC_Corpus/Entertainment_Music/test/formal" \
# --output_file "new_module/locate/locate_num_tokens_eda/GYAFC_Corpus_Entertainment_Music_test_informal_locate_max_5.txt" \
# --task formality \
# --label_id 0

# python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "/data/hyeryung/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/" \
# --input_file "data/formality/GYAFC_Corpus/Entertainment_Music/test/informal" \
# --output_file "new_module/locate/locate_num_tokens_eda/GYAFC_Corpus_Entertainment_Music_test_formal_locate_max_5.txt" \
# --task formality \
# --label_id 1

# python new_module/locate/new_locate_utils.py \
# --pretrained_model_path "/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
# --input_file "new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl" \
# --output_file "new_module/locate/locate_num_tokens_eda/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_nontoxic_locate_max_5.jsonl" \
# --task toxicity \
# --label_id 0

python new_module/locate/new_locate_utils.py \
--pretrained_model_path "/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
--input_file "new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl" \
--output_file "new_module/locate/locate_num_tokens_eda/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_nontoxic_locate_max_5_TEST.jsonl" \
--task toxicity \
--label_id 0


#### ==================== ####

# python new_module/locate/locate_utils_for_debug.py --method grad_norm \
# --input_path "new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
# --model_name_or_path /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_gn_refactored.jsonl" \
# --batch_size 8

# python new_module/locate/locate_utils.py --method attention \
# --input_path "new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
# --model_name_or_path /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_att.jsonl" \
# --batch_size 8

# python new_module/locate/locate_utils.py --method grad_norm \
# --input_path "new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
# --model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/' \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/testset_gpt2_2500_gn.jsonl" \
# --batch_size 8

# python new_module/locate/locate_utils.py --method attention \
# --input_path "new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
# --model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/' \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/testset_gpt2_2500_att.jsonl" \
# --batch_size 8

# python new_module/locate/locate_utils.py --method attention \
# --input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
# --model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/' \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_att.jsonl" \
# --batch_size 8

# python new_module/locate/locate_utils.py --method attention \
# --input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
# --model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint' \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_att.jsonl" \
# --batch_size 8

# python new_module/locate/locate_utils.py --method grad_norm \
# --input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
# --model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/' \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_gn.jsonl" \
# --batch_size 8

# python new_module/locate/locate_utils.py --method grad_norm \
# --input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
# --model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint' \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_gn.jsonl" \
# --batch_size 8