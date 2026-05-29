#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=n04
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=evaluate_llm_results      
#SBATCH --output='laser_edit/_slurm_outs/evaluate_llm_results_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO
results_dir="laser_edit/locate/llm/nli_toxicity/processed_results"
output_dir="laser_edit/locate/llm/nli_toxicity/evaluation_results"

python laser_edit/locate/llm/nli_toxicity/evaluate_llm_results.py \
--results_dir $results_dir \
--output_dir $output_dir