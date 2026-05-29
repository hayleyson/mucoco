#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --job-name=sc_energy_agent
#SBATCH --output='laser_edit/_slurm_outs/sc_energy_agent_%A_%a.out'

# SWEEP_ID is injected by the launcher via --export=ALL,SWEEP_ID=...
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID is not set. Run this script via ebm_edit_main_sc_energy_sweep.sh."
    exit 1
fi

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

echo "Array task ${SLURM_ARRAY_TASK_ID} starting agent for sweep ${SWEEP_ID}"

# Each agent picks up the next available run from the sweep queue.
wandb agent "${SWEEP_ID}"
