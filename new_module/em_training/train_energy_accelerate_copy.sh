#!/bin/bash
#SBATCH --time=0-72:00:00
#SBATCH --mem=250000MB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output='sentiment_model_%j.out'
#SBATCH --nodelist=master

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

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

# srun accelerate launch --num_processes=3 new_module/em_training/train_energy_model_copy.py \
# --model=roberta-base \
# --model_type=RobertaCustomForSequenceClassification \
# --batch_size=32 \
# --num_epochs=3 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=models/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training \
# --max_save_num=3 \
# --training_loss_type=cross_entropy \
# --val_loss_type=mse_loss \
# --train_data_path='data/yelp/train.jsonl' \
# --valid_data_path='data/yelp/valid.jsonl' \
# --wandb_project=sentiment \
# --task=sentiment \
# --num_validate_steps=100 

srun accelerate launch --num_processes=1 new_module/em_training/train_energy_model_copy.py \
--model=roberta-base \
--model_type=AutoModelForSequenceClassification \
--batch_size=16 \
--num_epochs=3 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-yelp-sentiment-classifier-energy-training \
--max_save_num=3 \
--training_loss_type=cross_entropy \
--val_loss_type=mse_loss \
--train_data_path='data/yelp/train.jsonl' \
--valid_data_path='data/yelp/valid.jsonl' \
--wandb_project=sentiment \
--task=sentiment \
--num_validate_steps=100 

