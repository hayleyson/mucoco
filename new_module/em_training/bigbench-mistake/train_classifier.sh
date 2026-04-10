#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodelist=n03
#SBATCH --mem=20gb
#SBATCH --output='new_module/_slurm_outs/train_classifier_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=$DATA_DIR/mucoco
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache

srun python new_module/em_training/bigbench-mistake/train_classifier.py\
 --data_dir new_module/data/BIG-Bench-Mistake\
 --train_full_20k\
 --output_base_dir new_module/em_training/bigbench-mistake/checkpoints\
 --wandb_project "mistake-finding-classifier"\
 --wandb_entity hayleyson\
 --model_name meta-llama/Llama-3.1-8B\
 --held_out_task logical_deduction\
 --max_length 8192
