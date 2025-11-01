#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --nodelist=n01
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=sc_energy_le
#SBATCH --output='new_module/_slurm_outs/sc_energy_le_%j.out'


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/.cache
export HF_DATASETS_CACHE=/data/hyeryung/.cache
export TRANSFORMERS_CACHE=/data/hyeryung/.cache
export LOGGING_LEVEL=DEBUG

srun python new_module/new_mlm_reranking_all_sc_energy.py \
vqa \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--losses gpt2 sc_energy \
--min_epsilons -1 \
--loss_weights 1 1 \
--k_per_location 5 \
--beam_size 5 \
--n_iter 8 \
--selection_criteria allsat_primary \
--locate_method grad_norm \
--wandb_project sc_energy \
--wandb_entity hayleyson