#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/eval_loc_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache

srun python new_module/locate/evaluate_locate.py \
--pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_gn.jsonl" \
--label_file_path="new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
--method="grad_norm" \
--dataset_type="gpt2" \
--tokenizer_path="/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/"

srun python new_module/locate/evaluate_locate.py \
--pred_file_path="new_module/data/toxicity-avoidance/testset_gpt2_2500_locate_grad.jsonl" \
--label_file_path="new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
--method="grad_norm" \
--dataset_type="gpt2" \
--tokenizer_path="/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/"

srun python new_module/locate/evaluate_locate.py \
--pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_att.jsonl" \
--label_file_path="new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
--method="attention" \
--dataset_type="gpt2" \
--tokenizer_path="/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/"

srun python new_module/locate/evaluate_locate.py \
--pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/testset_gpt2_2500_gn.jsonl" \
--label_file_path="new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
--method="grad_norm" \
--dataset_type="gpt2" \
--tokenizer_path="/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/"

srun python new_module/locate/evaluate_locate.py \
--pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/testset_gpt2_2500_att.jsonl" \
--label_file_path="new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl" \
--method="attention" \
--dataset_type="gpt2" \
--tokenizer_path="/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/"

# srun python new_module/locate/evaluate_locate.py \
# --pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_gn.jsonl" \
# --label_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_gn.jsonl" \
# --method="grad_norm" \
# --dataset_type="tsd"

# srun python new_module/locate/evaluate_locate.py \
# --pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_att.jsonl" \
# --label_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/tsd_test_att.jsonl" \
# --method="attention" \
# --dataset_type="tsd"

# srun python new_module/locate/evaluate_locate.py \
# --pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_gn.jsonl" \
# --label_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_gn.jsonl" \
# --method="grad_norm" \
# --dataset_type="tsd"

# srun python new_module/locate/evaluate_locate.py \
# --pred_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_att.jsonl" \
# --label_file_path="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/tsd_test_att.jsonl" \
# --method="attention" \
# --dataset_type="tsd"
