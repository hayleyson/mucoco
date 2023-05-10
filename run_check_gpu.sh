#!/bin/bash

#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --output='/home/hyeryungson/mucoco/slurm_output/%j.out'

source /home/${USER}/.bashrc
source ~/anaconda/etc/profile.d/conda.sh
conda activate mucoco

srun python check_gpu.py