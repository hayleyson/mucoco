#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:0
#SBATCH --output='laser_edit/_slurm_outs/calc_edited_only_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun -n 1 -c 1 python /home/hyeryung/data/mucoco/laser_edit/utils/dev_only/calc_editedsampleonly_metrics_original_gens.py \
--output_files /home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/scope/gpt2xl_1e-5_epoch20_zero_shot/scope_gen_nontoxic_scope_nontoxic_gpt2xl_1e-5_epoch20_zero_shot_20260417.jsonl \
--index_files laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332_index.txt \
--nicknames below_0_95 \
--task toxicity \
--sbert