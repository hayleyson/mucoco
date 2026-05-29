#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --output='laser_edit/_slurm_outs/optimize_locate_params_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python laser_edit/utils/dev_only/optimize_locate_params.py \
--output_dir outputs/sc_energy/set_nli/locate/eval2set_all/ \
--params_path laser_edit/set_consistency_energy/params_set_nli.yaml \
--dataset_name set_nli