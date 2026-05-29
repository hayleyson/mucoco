#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --nodelist=n01
#SBATCH --gres=gpu:1
#SBATCH --job-name=nli_energy
#SBATCH --output='laser_edit/_slurm_outs/####_%j.out'


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

git checkout nli
srun python laser_edit/ebm_training/nli/train_loss_mix.py 