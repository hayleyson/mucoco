#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --gres=gpu:0
#SBATCH --output='new_module/_slurm_outs/gbi_meta_%j.out'

sbatch new_module/decode_new_for_testset_iter_sentiment_em_le.sh
sbatch new_module/decode_new_for_testset_iter_sentiment_em_gn_le.sh
sbatch new_module/decode_new_for_testset_iter_sentiment_clsf_le.sh
sbatch new_module/decode_new_for_testset_iter_sentiment_clsf_gn_le.sh
sbatch new_module/decode_new_for_testset_iter_toxicity_em_le.sh
sbatch new_module/decode_new_for_testset_iter_toxicity_em_gn_le.sh
sbatch new_module/decode_new_for_testset_iter_toxicity_clsf_le.sh
sbatch new_module/decode_new_for_testset_iter_toxicity_clsf_gn_le.sh
sbatch new_module/decode_new_for_testset_iter_formality_em_le.sh
sbatch new_module/decode_new_for_testset_iter_formality_em_gn_le.sh
sbatch new_module/decode_new_for_testset_iter_formality_clsf_le.sh
sbatch new_module/decode_new_for_testset_iter_formality_clsf_gn_le.sh

