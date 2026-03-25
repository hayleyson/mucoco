#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --nodelist=n01
#SBATCH --output='new_module/_slurm_outs/set_consistency_edit_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export OPENAI_API_KEY=

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export HF_DATASETS_CACHE=/home/hyeryung/data/hf_cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/hf_cache

# srun -n 1 -c 1 python new_module/set_consistency_llm_edit.py \
# Qwen/Qwen2.5-7B-Instruct \
# --output_dir new_module/_notebooks/20260309_UIUC_talk_outputs/testset_incon_2/ \
# --use_incon_samples \
# --n_samples 2 \
# --config_path new_module/set_consistency_energy/params_set_lconvqa.yaml

srun -n 1 -c 1 python new_module/set_consistency_llm_edit.py \
gpt-5.4 \
--output_dir new_module/_notebooks/20260309_UIUC_talk_outputs/testset_incon_300/ \
--use_incon_samples \
--n_samples 300 \
--reasoning_effort medium \
--config_path new_module/set_consistency_energy/params_set_lconvqa.yaml \
--mode wo_locate