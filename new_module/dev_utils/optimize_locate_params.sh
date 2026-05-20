#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --output='new_module/_slurm_outs/optimize_locate_params_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python new_module/dev_utils/optimize_locate_params.py \
--output_dir outputs/sc_energy/set_lconvqa/locate/eval2set_all/clsf \
--params_path new_module/set_consistency_energy/params_set_lconvqa_clsf.yaml