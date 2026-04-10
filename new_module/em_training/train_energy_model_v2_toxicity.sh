#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --mem=15GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/toxicity_model_%j.out'
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

# srun accelerate launch --num_processes=4 new_module/em_training/train_energy_model_v2_resume.py \
# --model=roberta-base-custom \
# --model_type=RobertaCustomForSequenceClassification \
# --batch_size=56 \
# --num_epochs=3 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-binary-dump-final \
# --max_save_num=3 \
# --training_loss_type=cross_entropy \
# --val_loss_type=mse_loss \
# --train_data_path='/home/hyeryung/data/loc_edit/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/train_bin_dump.jsonl' \
# --valid_data_path='/home/hyeryung/data/loc_edit/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/valid_bin_dump.jsonl' \
# --wandb_project=toxicity \
# --task=toxicity \
# --num_validate_steps=100 \
# --resume_wandb_id=yd6b3nbx \
# --resume_from_checkpoint=""

srun accelerate launch --num_processes=1 new_module/em_training/train_energy_model_v2_resume.py \
--model=roberta-base-custom \
--model_type=RobertaCustomForSequenceClassification \
--batch_size=56 \
--num_epochs=3 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-with-gemma2-embeds \
--max_save_num=1 \
--training_loss_type=cross_entropy \
--val_loss_type=cross_entropy \
--train_data_path='/home/hyeryung/data/mucoco/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/train_binary.jsonl' \
--valid_data_path='/home/hyeryung/data/mucoco/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/dev_binary.jsonl' \
--wandb_project=toxicity \
--task=toxicity \
--num_validate_steps=100

