#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --output='sentiment_model_%j.out'
#SBATCH --nodelist=n02

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit


DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# srun accelerate launch --num_processes=3 examples/training_constraint_models/train_energy_model.py \
# --model=custom-roberta-base-regressor \
# --batch_size=16 \
# --num_epochs=3 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --margin=0.0 \
# --checkpoint_path=models/roberta-base-yelp-sentiment-regressor-with-gpt2-large-embeds-rescale \
# --max_save_num=3 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.00 \
# --train_data_path='data/yelp/train.jsonl' \
# --valid_data_path='data/yelp/valid.jsonl' \
# --wandb_project=sentiment \
# --task=sentiment \
# --num_validate_steps=100 

# srun accelerate launch --num_processes=3 examples/training_constraint_models/train_energy_model.py \
# --model=roberta-base \
# --model_type=AutoModelForSequenceClassification \
# --batch_size=32 \
# --num_epochs=3 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=models/roberta-base-yelp-sentiment-classifier-energy-training \
# --max_save_num=3 \
# --training_loss_type=cross_entropy \
# --val_loss_type=mse_loss \
# --train_data_path='data/yelp/train.jsonl' \
# --valid_data_path='data/yelp/valid.jsonl' \
# --wandb_project=sentiment \
# --task=sentiment \
# --num_validate_steps=100 

srun accelerate launch --num_processes=2 new_module/em_training/train_energy_model_v2_resume.py \
--model=roberta-base \
--model_type=RobertaCustomForSequenceClassification \
--batch_size=32 \
--num_epochs=3 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=/data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds \
--max_save_num=3 \
--training_loss_type=cross_entropy \
--val_loss_type=mse_loss \
--train_data_path='data/yelp/train_binary.jsonl' \
--valid_data_path='data/yelp/valid_binary.jsonl' \
--wandb_project=sentiment \
--task=sentiment \
--num_validate_steps=100 

