#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/em4mlm_eval_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit



export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export HF_DATASETS_CACHE=/home/hyeryung/data/hf_cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO


srun examples/prompt/toxicity-all/mucola-disc-save-init-gen.sh

