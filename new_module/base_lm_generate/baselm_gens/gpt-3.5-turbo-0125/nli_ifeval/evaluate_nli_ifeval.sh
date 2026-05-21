#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=eval_nli_ifeval
#SBATCH --output='new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun python new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/evaluate_nli_ifeval.py