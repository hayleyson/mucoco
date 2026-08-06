#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=plain_lconvqa
#SBATCH --output='laser_edit/_slurm_outs/plain_llm_lconvqa_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: Plain LLM edit | Task: set-LConVQA
# Source: llm_edit_main_sc_energy_instance.sh + qwen25_lconvqa_edit_nucleus_rerun.sh (wo_locate)

srun -n 1 -c 2 python laser_edit/llm_edit_main_sc_energy_instance.py \
Qwen/Qwen2.5-7B-Instruct \
--output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
--use_incon_samples \
--n_samples 300 \
--mode wo_locate \
--dataset_name lconvqa \
--decoding nucleus \
--random_seed 42
