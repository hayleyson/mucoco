#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --nodelist=n01
#SBATCH --output='laser_edit/_slurm_outs/set_consistency_locate_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export OPENAI_API_KEY=

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

# srun python laser_edit/processbench_critique/run_llm_locate_gpt.py --configs prm800k --model_name gpt-5-mini

srun python laser_edit/processbench_critique/run_llm_locate.py --configs prm800k --model_path Qwen/Qwen3-8B