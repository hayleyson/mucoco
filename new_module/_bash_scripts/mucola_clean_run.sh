#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-18:00:00
#SBATCH --mem=10GB
#SBATCH --nodelist=n01
#SBATCH --gres=gpu:1
#SBATCH --job-name=mucola_formal_decode
#SBATCH --output='new_module/_slurm_outs/mucola_formal_decode_%j.out'


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun python decode_new_clean.py --argument_file_path examples/prompt/toxicity-all/arguments_below_nontoxic_threshold_468.txt
srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_negative_threshold_827.txt
srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_positive_threshold_778.txt