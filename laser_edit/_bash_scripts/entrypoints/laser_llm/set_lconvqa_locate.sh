#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=laser_llm_lconvqa_loc
#SBATCH --output='laser_edit/_slurm_outs/laser_llm_lconvqa_locate_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: LASER & LLM edit | Task: set-LConVQA | Step: EBM/LASER locate
# Source: qwen25_lconvqa_locate_rerun.sh (ebm)

srun -n 1 -c 2 python laser_edit/locate/ebm/set_consistency_locate_instance.py \
ebm \
--dataset_name lconvqa \
--output_dir outputs/sc_energy/set_lconvqa/locate/testset_incon_300 \
--use_incon_samples \
--n_samples 300 \
--random_seed 42 \
--ebm_config_path laser_edit/set_consistency_energy/params_set_lconvqa.yaml
