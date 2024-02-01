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

# sbatch new_module/mlm_reranking_toxicity_original_meta.sh
# sbatch new_module/mlm_reranking_toxicity_v0_meta.sh
sbatch new_module/mlm_reranking_toxicity_v1_meta.sh
sbatch new_module/mlm_reranking_toxicity_v2_meta.sh

# sbatch new_module/mlm_reranking_formality_original_meta.sh
# sbatch new_module/mlm_reranking_formality_v0_meta.sh
sbatch new_module/mlm_reranking_formality_v1_meta.sh
sbatch new_module/mlm_reranking_formality_v2_meta.sh

# sbatch new_module/mlm_reranking_sentiment_original_meta.sh
# sbatch new_module/mlm_reranking_sentiment_v0_meta.sh
sbatch new_module/mlm_reranking_sentiment_v1_meta.sh
sbatch new_module/mlm_reranking_sentiment_v2_meta.sh

