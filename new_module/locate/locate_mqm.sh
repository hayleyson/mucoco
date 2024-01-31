#!/bin/bash
#SBATCH --job-name=mqm-locate
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=8
#SBATCH --output='mqm_locate_%j.out'

start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unbabel3.8

srun python locate_mqm.py