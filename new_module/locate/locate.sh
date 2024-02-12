#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/loc_skiml_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache

# python new_module/locate/locate_utils.py --method grad_norm \
# --input_path "new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
# --model_name_or_path /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint \
# --model_type RobertaCustomForSequenceClassification \
# --output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_gn.jsonl" \
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

python new_module/locate/locate_utils.py --method attention \
--input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
--model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/' \
--model_type RobertaCustomForSequenceClassification \
--output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_att.jsonl" \
--batch_size 8

python new_module/locate/locate_utils.py --method attention \
--input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
--model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint' \
--model_type RobertaCustomForSequenceClassification \
--output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_att.jsonl" \
--batch_size 8

python new_module/locate/locate_utils.py --method grad_norm \
--input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
--model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/' \
--model_type RobertaCustomForSequenceClassification \
--output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_gn.jsonl" \
--batch_size 8

python new_module/locate/locate_utils.py --method grad_norm \
--input_path "data/toxic_spans/SemEval2021/data/tsd_test.csv" \
--model_name_or_path '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint' \
--model_type RobertaCustomForSequenceClassification \
--output_path "new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_gn.jsonl" \
--batch_size 8