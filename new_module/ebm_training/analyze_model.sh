#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --output='new_module/ebm_training/analyze_model_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

# srun python new_module/ebm_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier/step_500_best_checkpoint/\
#  --model_type roberta-base\
#  --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/valid.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/ebm_training/toxicity/evaluation_results/roberta-base-jigsaw-toxicity-classifier/analysis/valid\
#  --batch_size 32

# srun python new_module/ebm_training/analyze_model.py\
#  --checkpoint_dir /home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/\
#  --model_file_name best_model_pearsonr.pth\
#  --model_type encoder-model\
#  --test_data_path data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl\
#  --test_data_type valid\
#  --output_dir new_module/ebm_training/nli/evaluation_results/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/valid\
#  --batch_size 32\
#  --task nli
