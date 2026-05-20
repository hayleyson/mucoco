#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=40gb
#SBATCH --output='new_module/ebm_training/find_threshold_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache


# # Find out best thresholds
# srun python new_module/ebm_training/find_classification_threshold.py \
# --model_path /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
# --validation_dataset_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/valid.jsonl \
# --task toxicity \
# --label_id 0 \
# --batch_size 64 \
# --num_workers 2
# --save_testset_edit_candidates \
# --testset_path new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl \
# --threshold 0.95

# Save data according to the threshold
# srun python new_module/ebm_training/find_classification_threshold.py \
# --model_path /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
# --validation_dataset_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/valid.jsonl \
# --task toxicity \
# --label_id 0 \
# --batch_size 64 \
# --num_workers 2 \
# --save_testset_edit_candidates \
# --testset_path new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl \
# --threshold 0.29

# # Find out best thresholds
# srun python new_module/ebm_training/find_classification_threshold.py \
# --model_path /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/ \
# --validation_dataset_path data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl \
# --task nli \
# --label_id 1 \
# --batch_size 64 \
# --num_workers 2

# # Save data according to the threshold
# srun python new_module/ebm_training/find_classification_threshold.py \
# --model_path /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/ \
# --validation_dataset_path data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl \
# --task nli \
# --label_id 1 \
# --batch_size 64 \
# --num_workers 2 \
# --save_testset_edit_candidates \
# --testset_path new_module/data/logical-consistency/anli-r2-test_prompt_4.jsonl \
# --threshold 0.02

srun -n 1 -c 1 python new_module/ebm_training/find_classification_threshold.py \
--model_path /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/ \
--validation_dataset_path /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150_postprocessed.jsonl \
--task nli \
--label_id 1 \
--batch_size 32 \
--num_workers 2 \
--save_testset_edit_candidates \
--testset_path /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150_postprocessed.jsonl \
--threshold 0.99