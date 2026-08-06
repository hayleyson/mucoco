#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=sl_snli_edit
#SBATCH --output='laser_edit/_slurm_outs/self_loc_nli_set_edit_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: Self-locate & LLM edit | Task: set-NLI (set-SNLI) | Step: edit
# Requires: self_locate_llm/set_nli_locate.sh outputs under locate/

srun -n 1 -c 2 python laser_edit/llm_edit_main_sc_energy_instance.py \
Qwen/Qwen2.5-7B-Instruct \
--output_dir outputs/sc_energy/set_nli/llm/testset_incon_300/ \
--use_incon_samples \
--n_samples 300 \
--mode w_self_locate \
--dataset_name set_nli \
--decoding nucleus \
--random_seed 42
