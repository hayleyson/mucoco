#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --mem=15GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --output='new_module/_slurm_outs/irony_model_%j.out'
#SBATCH --nodelist=n01


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun accelerate launch --num_processes=2 new_module/em_training/train_energy_model_v2_resume.py \
--model=roberta-base \
--model_type=AutoModelForSequenceClassification \
--batch_size=56 \
--num_epochs=3 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=/data/hyeryung/loc_edit/models/roberta-base-irony-classifier-balanced-training \
--max_save_num=1 \
--training_loss_type=cross_entropy \
--val_loss_type=cross_entropy \
--train_data_path='data/ACL-2014-irony/train_binary.jsonl' \
--valid_data_path='data/ACL-2014-irony/val_binary.jsonl' \
--wandb_project=irony-energy-model \
--task=irony \
--num_validate_steps=50 \
--balanced_sampling



# srun accelerate launch --num_processes=2 --main_process_port=29876 new_module/em_training/train_energy_model_v2_resume.py \
# --model=roberta-base \
# --model_type=AutoModelForSequenceClassification \
# --batch_size=56 \
# --num_epochs=3 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=/data/hyeryung/loc_edit/models/roberta-base-irony-energy-model-balanced-training \
# --max_save_num=1 \
# --training_loss_type=cross_entropy \
# --val_loss_type=cross_entropy \
# --train_data_path='data/ACL-2014-irony/train.jsonl' \
# --valid_data_path='data/ACL-2014-irony/val.jsonl' \
# --wandb_project=irony-energy-model \
# --task=irony \
# --num_validate_steps=50 \
# --balanced_sampling