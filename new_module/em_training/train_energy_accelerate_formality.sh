#!/bin/bash
#SBATCH --time=0-72:00:00
#SBATCH --mem=250000MB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output='formality_model_%j.out'
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

# srun accelerate launch --num_processes=1 examples/training_constraint_models/train_energy_model_copy.py \
# --model=roberta-base \
# --model_type=AutoModelForSequenceClassification \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=models/roberta-base-pt16-formality-classifier-energy-training \
# --max_save_num=2 \
# --training_loss_type=cross_entropy \
# --val_loss_type=mse_loss \
# --train_data_path='data/formality/PT16/train.tsv' \
# --valid_data_path='data/formality/PT16/valid.tsv' \
# --wandb_project=formality \
# --task=formality \
# --num_validate_steps=100 

srun accelerate launch --num_processes=1 examples/training_constraint_models/train_energy_model_copy.py \
--model=roberta-base \
--model_type=AutoModelForSequenceClassification \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-regressor-rescale \
--max_save_num=2 \
--training_loss_type=mse \
--val_loss_type=mse_loss \
--loss_weight_ranking=0.00 \
--train_data_path='data/formality/PT16/train.tsv' \
--valid_data_path='data/formality/PT16/valid.tsv' \
--wandb_project=formality \
--task=formality \
--num_validate_steps=100 