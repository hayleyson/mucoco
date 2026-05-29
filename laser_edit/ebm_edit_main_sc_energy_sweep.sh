#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=4GB
#SBATCH --nodelist=master
#SBATCH --job-name=sc_energy_sweep_launch
#SBATCH --output='laser_edit/_slurm_outs/sc_energy_sweep_launch_%j.out'

# Number of parallel wandb agents to run simultaneously.
# Each agent picks the next available run from the sweep queue,
# so N_AGENTS jobs will collectively cover all sweep combinations.
N_AGENTS=2
WANDB_PROJECT=set-nli-decoding
WANDB_ENTITY=hayleyson

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

# ---------------------------------------------------------------------------
# Step 1: Register the sweep and capture the sweep ID.
# ---------------------------------------------------------------------------
SWEEP_OUTPUT=$(wandb sweep \
    --project "${WANDB_PROJECT}" \
    --entity "${WANDB_ENTITY}" \
    laser_edit/ebm_edit_main_sc_energy_sweep.yaml 2>&1)
echo "$SWEEP_OUTPUT"

SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP "(?<=wandb agent )[\w/\-]+")

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: could not parse sweep ID from wandb output. Aborting."
    exit 1
fi

echo "Registered sweep: $SWEEP_ID"

# ---------------------------------------------------------------------------
# Step 2: Submit an array of agent jobs, each running `wandb agent`.
#         SWEEP_ID is exported so the agent script can read it.
# ---------------------------------------------------------------------------
sbatch \
    --array=1-${N_AGENTS} \
    --export=ALL,SWEEP_ID=${SWEEP_ID} \
    laser_edit/ebm_edit_main_sc_energy_agent.sh

echo "Submitted ${N_AGENTS} agent jobs for sweep ${SWEEP_ID}."
