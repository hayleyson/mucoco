#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --output='laser_edit/_slurm_outs/set_consistency_locate_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm


export OPENAI_API_KEY=
export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

export LOGGING_LEVEL=DEBUG

srun -n 1 -c 1 python laser_edit/locate/llm/set_consistency/set_consistency_locate.py \
Qwen/Qwen2.5-7B-Instruct \
--dataset_name set_lconvqa \
--use_incon_samples \
--n_samples 300 \
--output_dir outputs/sc_energy/set_lconvqa/locate/testset_incon_300