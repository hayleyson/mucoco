#!/bin/bash
#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output='slurm_output/%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

jobid=$(sbatch --parsable new_module/mlm_reranking_sentiment_original_em.sh True)

dependency="afterany:${jobid}"
jobid=$(sbatch --parsable --dependency=$dependency new_module/mlm_reranking_sentiment_original_clsf.sh True)
