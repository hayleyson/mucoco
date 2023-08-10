#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=cas_v100_2
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --output=/scratch/x2600a03/mucoco/jupyter.log
#SBATCH --comment jupyter

module load python/3.9.5
conda init bash
conda activate mucola

jupyter lab --no-browser --port=8888