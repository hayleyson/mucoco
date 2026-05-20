#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=master
#SBATCH --output='new_module/_slurm_outs/set_consistency_edit_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

srun -n 1 -c 1 python new_module/set_consistency_llm_edit.py \
Qwen/Qwen2.5-7B-Instruct \
--output_dir outputs/sc_energy/set_nli/llm/testset_incon_300/ \
--use_incon_samples \
--n_samples 300 \
--mode w_self_locate \
--dataset_name set_nli

# srun -n 1 -c 1 python new_module/set_consistency_llm_edit.py \
# Qwen/Qwen3-8B \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --use_incon_samples \
# --n_samples 300 \
# --reasoning_effort medium \
# --mode w_ebm_locate

# srun -n 1 -c 1 python new_module/set_consistency_llm_edit.py \
# gpt-5.4 \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --use_incon_samples \
# --n_samples 300 \
# --reasoning_effort none \
# --mode w_ebm_locate

# srun -n 1 -c 1 python new_module/set_consistency_llm_edit.py \
# gpt-5-mini \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --use_incon_samples \
# --n_samples 300 \
# --reasoning_effort medium \
# --mode w_ebm_locate