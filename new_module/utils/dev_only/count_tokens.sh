#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --nodelist=n02
#SBATCH --output='new_module/_slurm_outs/count_tokens_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python new_module/utils/count_tokens.py \
--input_file /home/hyeryung/data/mucoco/outputs/sc_energy/set_lconvqa/ebm/pqg6o3gb/outputs_Qwen2.5-7B-Instruct_v1_s0_p0.96_refined.jsonl \
--model_name Qwen/Qwen2.5-7B-Instruct