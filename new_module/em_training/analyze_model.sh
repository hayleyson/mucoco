#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --output='new_module/em_training/analyze_model_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export HF_DATASETS_CACHE=/home/hyeryung/data/hf_cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/hf_cache

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint/\
#  --model_type roberta-base\
#  --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/valid.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-jigsaw-toxicity-classifier-energy-training/valid\
#  --batch_size 32

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/\
#  --model_type roberta-base\
#  --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/valid.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-jigsaw-toxicity-classifier/analysis/valid\
#  --batch_size 32

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint\
#  --model_type roberta-base\
#  --test_data_path data/yelp/valid.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-yelp-sentiment-classifier-energy-training/valid\
#  --batch_size 32

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/roberta-base-yelp-sentiment-classifier/step_83000\
#  --model_type roberta-base\
#  --test_data_path data/yelp/valid.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-yelp-sentiment-classifier/valid\
#  --batch_size 32

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/\
#  --model_file_name best_model_pearsonr.pth\
#  --model_type encoder-model\
#  --test_data_path data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/valid\
#  --batch_size 32\
#  --task nli

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443/\
#  --model_file_name best_model_pearsonr.pth\
#  --model_type encoder-model\
#  --test_data_path data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/valid\
#  --batch_size 32\
#  --task nli

srun python new_module/em_training/analyze_model.py\
 --checkpoint_dir /home/hyeryung/data/loc_edit/models/roberta-base-irony-energy-model/step_200_best_checkpoint\
 --model_file_name model.safetensors\
 --model_type roberta-base\
 --test_data_path data/ACL-2014-irony/test_binary.jsonl\
 --test_data_type test\
 --output_dir new_module/em_training/roberta-base-irony-energy-model/test\
 --batch_size 32\
 --task irony

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/roberta-base-irony-classifier/step_200_best_checkpoint\
#  --model_file_name model.safetensors\
#  --model_type roberta-base\
#  --test_data_path data/ACL-2014-irony/test_binary.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-irony-classifier/test\
#  --batch_size 32\
#  --task irony