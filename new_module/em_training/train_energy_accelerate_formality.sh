#!/bin/bash
#SBATCH --time=0-72:00:00
#SBATCH --mem=250000MB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output='formality_model_%j.out'
#SBATCH --nodelist=n02

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

# srun accelerate launch --config ~/.cache/huggingface/accelerate/default_config.yaml --num_processes=1 new_module/em_training/train_energy_model_update.py \
# --model=roberta-base-custom \
# --model_type=RobertaCustomForSequenceClassification \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=models/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training \
# --max_save_num=1 \
# --training_loss_type=cross_entropy \
# --val_loss_type=mse_loss \
# --loss_weight_ranking=0.00 \
# --train_data_path='data/formality/PT16/train.tsv' \
# --valid_data_path='data/formality/PT16/valid.tsv' \
# --wandb_project=formality \
# --task=formality \
# --num_validate_steps=-1 

# srun accelerate launch --config ~/.cache/huggingface/accelerate/default_config.yaml --num_processes=1 new_module/em_training/train_energy_model_update.py \
# --model=roberta-base \
# --model_type=AutoModelForSequenceClassification \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=models/roberta-base-pt16-formality-classifier-energy-training \
# --max_save_num=1 \
# --training_loss_type=cross_entropy \
# --val_loss_type=mse_loss \
# --loss_weight_ranking=0.00 \
# --train_data_path='data/formality/PT16/train.tsv' \
# --valid_data_path='data/formality/PT16/valid.tsv' \
# --wandb_project=formality \
# --task=formality \
# --num_validate_steps=-1 

srun accelerate launch --num_processes=1 new_module/em_training/train_energy_model_update.py \
--model=roberta-base \
--model_type=AutoModelForSequenceClassification \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-classifier \
--max_save_num=1 \
--training_loss_type=cross_entropy \
--val_loss_type=mse_loss \
--loss_weight_ranking=0.00 \
--train_data_path='data/formality/PT16/train_binary.tsv' \
--valid_data_path='data/formality/PT16/valid_binary.tsv' \
--wandb_project=formality \
--task=formality \
--num_validate_steps=-1 

srun accelerate launch --num_processes=1 new_module/em_training/train_energy_model_update.py \
--model=roberta-base-custom \
--model_type=RobertaCustomForSequenceClassification \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds \
--max_save_num=1 \
--training_loss_type=cross_entropy \
--val_loss_type=mse_loss \
--loss_weight_ranking=0.00 \
--train_data_path='data/formality/PT16/train_binary.tsv' \
--valid_data_path='data/formality/PT16/valid_binary.tsv' \
--wandb_project=formality \
--task=formality \
--num_validate_steps=-1 