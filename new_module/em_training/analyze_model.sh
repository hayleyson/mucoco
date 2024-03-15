#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --partition=P1
#SBATCH --output='new_module/em_training/analyze_model_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache


# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds\
#  --model_type roberta-base-custom\
#  --test_data_path data/formality/PT16/valid.tsv\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds/analysis/valid

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training\
#  --model_type roberta-base-custom\
#  --test_data_path data/formality/PT16/valid.tsv\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training/analysis/valid

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier\
#  --model_type roberta-base\
#  --test_data_path data/formality/PT16/valid.tsv\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier/analysis/valid

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-energy-training\
#  --model_type roberta-base\
#  --test_data_path data/formality/PT16/valid.tsv\
#  --test_data_type valid\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-energy-training/analysis/valid\

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-gyafc\
#  --model_type roberta-base\
#  --test_data_path data/formality/GYAFC_Corpus/Entertainment_Music/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-gyafc/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc\
#  --model_type roberta-base-custom\
#  --test_data_path data/formality/GYAFC_Corpus/Entertainment_Music/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds\
#  --model_type roberta-base-custom\
#  --test_data_path data/formality/GYAFC_Corpus/Entertainment_Music/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training\
#  --model_type roberta-base-custom\
#  --test_data_path data/formality/GYAFC_Corpus/Entertainment_Music/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier\
#  --model_type roberta-base\
#  --test_data_path data/formality/GYAFC_Corpus/Entertainment_Music/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-energy-training\
#  --model_type roberta-base\
#  --test_data_path data/formality/GYAFC_Corpus/Entertainment_Music/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-energy-training/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc\
#  --model_type roberta-base-custom\
#  --test_data_path data/formality/PT16/valid.tsv\
#  --test_data_type pt16_valid\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/analysis/pt16_valid

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-gyafc\
#  --model_type roberta-base\
#  --test_data_path data/formality/PT16/valid.tsv\
#  --test_data_type pt16_valid\
#  --output_dir new_module/em_training/roberta-base-pt16-formality-classifier-gyafc/analysis/pt16_valid

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier\
#  --model_type roberta-base\
#  --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-jigsaw-toxicity-classifier/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-energy-training\
#  --model_type roberta-base\
#  --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-jigsaw-toxicity-classifier-energy-training/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds\
#  --model_type roberta-base-custom\
#  --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training\
#  --model_type roberta-base-custom\
#  --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/analysis/test

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training\
#  --model_type roberta-base-custom\
#  --test_data_path /shared/s3/lab07/hyeryung/yelp/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/analysis/test\
#  --batch_size 32

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds\
#  --model_type roberta-base-custom\
#  --test_data_path /shared/s3/lab07/hyeryung/yelp/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds/analysis/test\
#  --batch_size 32

srun python new_module/em_training/analyze_model.py\
 --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/models_re/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint/\
 --model_type roberta-base\
 --test_data_path data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/test.jsonl\
 --test_data_type test\
 --output_dir new_module/em_training/roberta-base-jigsaw-toxicity-classifier-energy-training/analysis/test\
 --batch_size 32

# srun python new_module/em_training/analyze_model.py\
#  --checkpoint_dir /shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds\
#  --model_type roberta-base-custom\
#  --test_data_path /shared/s3/lab07/hyeryung/yelp/test.jsonl\
#  --test_data_type test\
#  --output_dir new_module/em_training/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds/analysis/test\
#  --batch_size 32