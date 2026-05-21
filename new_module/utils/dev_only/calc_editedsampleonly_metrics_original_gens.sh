#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:0
#SBATCH --job-name=###
#SBATCH --output='new_module/_slurm_outs/####_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/utils/calc_editedsampleonly_metrics_original_gens.py \
--output_files /home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/scope/gpt2xl_1e-5_epoch20_edg_clsf/scope_gen_nontoxic_scope_nontoxic_gpt2xl_1e-5_epoch20_edg_clsf_plain_20260417.jsonl \
--index_files new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332_index.txt \
--nicknames below_0_95 \
--task toxicity