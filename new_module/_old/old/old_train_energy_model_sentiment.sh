#!/bin/bash
#SBATCH --job-name=sentiment-ranker
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=50gb
#SBATCH --partition=P1
#SBATCH --output='slurm_output/%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python examples/training_constraint_models/train_energy_model.py \
--model=custom-roberta-base-regressor \
--batch_size=16 \
--num_epochs=5 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--margin=0.0 \
--checkpoint_path=models/roberta-base-yelp-sentiment-regressor-with-gpt2-large-embeds-rescale \
--max_save_num=1 \
--val_loss_type=mse_loss \
--loss_weight_mse=1.0 \
--loss_weight_ranking=0.00 \
--train_data_path='../data/yelp/train.jsonl' \
--valid_data_path='../data/yelp/valid.jsonl' \
--wandb_project=sentiment \
--task=sentiment


# srun python examples/training_constraint_models/train_energy_model.py \
# --model=custom-roberta-base-regressor \
# --batch_size=16 \
# --num_epochs=5 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --margin=0.0 \
# --checkpoint_path=models/roberta-base-yelp-sentiment-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss \
# --max_save_num=1 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.05 \
# --ranking_loss_type=scaled_ranking_loss \
# --train_data_path='../data/yelp/train.jsonl' \
# --valid_data_path='../data/yelp/valid.jsonl' \
# --wandb_project=sentiment \
# --task=sentiment
