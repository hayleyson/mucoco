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
export HF_DATASETS_CACHE=/home/hyeryung/data/hf_cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python new_module/dev_utils/count_tokens.py --input_file /home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/ebm/j18pi8ab/final/outputs_epsilon0.95.txt.0_qwen2.5_7B_s0_p0.96_refined_initial_prompt.jsonl --model_name Qwen/Qwen2.5-7B-Instruct