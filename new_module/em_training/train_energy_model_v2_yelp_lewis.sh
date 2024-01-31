#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --mem=160GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output='new_module/em_training/sentiment_lewis_model_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun accelerate launch --num_processes=2 new_module/em_training/train_energy_model_v2_resume.py \
--model=roberta-base-custom \
--model_type=RobertaCustomForSequenceClassification \
--batch_size=256 \
--num_epochs=3 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=/data/hyeryung/loc_edit/models/roberta-base-yelp-lewis-sentiment-classifier-with-gpt2-large-embeds-binary \
--max_save_num=3 \
--training_loss_type=cross_entropy \
--val_loss_type=mse_loss \
--train_data_path='data/sentiment/language-style-transfer/data/yelp_for_training/train.jsonl' \
--valid_data_path='data/sentiment/language-style-transfer/data/yelp_for_training/valid.jsonl' \
--wandb_project=sentiment \
--task=sentiment \
--num_validate_steps=100

