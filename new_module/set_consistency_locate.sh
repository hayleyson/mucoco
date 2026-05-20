#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/set_consistency_locate_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm


export OPENAI_API_KEY=
export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

export LOGGING_LEVEL=DEBUG

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# gpt-5-mini \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --reasoning_effort medium \
# --use_incon_samples \
# --n_samples 300

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# gpt-5.4 \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --reasoning_effort none \
# --use_incon_samples \
# --n_samples 300

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# Qwen/Qwen3-8B \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --use_incon_samples \
# --n_samples 300

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# Qwen/Qwen2.5-7B-Instruct \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --use_incon_samples \
# --n_samples 300

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# ebm \
# --output_dir outputs/sc_energy/set_lconvqa/locate/testset_all/ \
# --ebm_config_path new_module/set_consistency_energy/params_set_lconvqa.yaml

srun -n 1 -c 1 python new_module/set_consistency_locate.py \
Qwen/Qwen2.5-7B-Instruct \
--dataset_name set_nli \
--output_dir outputs/sc_energy/set_nli/locate/testset_incon_300 \
--use_incon_samples \
--n_samples 300

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# classifier \
# --output_dir outputs/sc_energy/set_lconvqa/locate/testset_incon_300 \
# --ebm_config_path new_module/set_consistency_energy/params_set_lconvqa_clsf.yaml \
# --use_incon_samples \
# --n_samples 300

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# gpt-5.4 \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --dataset_path new_module/data/convqa/processed_data/lconvqa_test1_incon_300.pickle \
# --reasoning_effort none

# srun python new_module/set_consistency_locate.py \
# gpt-5-mini \
# --output_dir outputs/sc_energy/set_lconvqa/locate/testset_all

# srun python new_module/set_consistency_locate.py \
# gpt-4o \
# --output_dir outputs/sc_energy/set_lconvqa/locate/testset_all

# srun python new_module/set_consistency_locate.py \
# Qwen/Qwen3-8B \
# --output_dir outputs/sc_energy/set_lconvqa/locate/testset_all

# srun python new_module/set_consistency_locate.py \
# Qwen/QwQ-32B-Preview \
# --output_dir outputs/sc_energy/set_lconvqa/locate/testset_all

# srun python new_module/set_consistency_locate.py \
# gpt-5.4 \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300 \
# --use_incon_samples \
# --n_samples 300 \
# --random_seed 42

# srun -n 1 -c 1 python new_module/set_consistency_locate.py \
# gpt-5.4 \
# --output_dir outputs/sc_energy/set_lconvqa/locate/testset_all \
# --reasoning_effort none