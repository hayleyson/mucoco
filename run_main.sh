#!/bin/bash

#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=0
#SBATCH --partition=P1
#SBATCH --output='/home/hyeryungson/mucoco/slurm_output/%j.out'

echo "hello";