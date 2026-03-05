#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --nodelist=n01
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=sc_energy_le
#SBATCH --output='new_module/_slurm_outs/sc_locate_params_%j.out'


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export HF_DATASETS_CACHE=/home/hyeryung/data/.cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

# ----------------------------------------------------------------------- #
# Set-LconVQA (CLSF)
# ----------------------------------------------------------------------- #
## Attention - explore layers & agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa_clsf.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/lcon_set_sup_2-1213354 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa_clsf.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/lcon_set_sup_2-1213354 \
--device cuda \
--agg_method median

## Grad Norm - explore agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa_clsf_gradnorm.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/lcon_set_sup_2-1213354 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa_clsf_gradnorm.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/lcon_set_sup_2-1213354 \
--device cuda \
--agg_method median

# ----------------------------------------------------------------------- #
# Set-SNLI (CLSF)
# ----------------------------------------------------------------------- #
## Attention - Explore layers & agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli_clsf.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/nli_set_sup_2-1213401 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli_clsf.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/nli_set_sup_2-1213401 \
--device cuda \
--agg_method median

## Grad Norm - explore agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli_clsf_gradnorm.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/nli_set_sup_2-1213401 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli_clsf_gradnorm.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/nli_set_sup_2-1213401 \
--device cuda \
--agg_method median


# ----------------------------------------------------------------------- #
# Set-LconVQA (EBM)
# ----------------------------------------------------------------------- #
## Attention - explore layers & agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/1225068 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/1225068 \
--device cuda \
--agg_method median

## Grad Norm - explore agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa_gradnorm.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/1225068 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_lconvqa \
--config new_module/set_consistency_energy/params_set_lconvqa_gradnorm.yaml \
--data_path new_module/data/convqa/eval2/set_lconvqa_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/vqa/lconvqa/1225068 \
--device cuda \
--agg_method median

# ----------------------------------------------------------------------- #
# Set-SNLI (EBM)
# ----------------------------------------------------------------------- #
## Attention - Explore layers & agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/46853 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/46853 \
--device cuda \
--agg_method median

## Grad Norm - explore agg method
srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli_gradnorm.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/46853 \
--device cuda \
--agg_method avg

srun python new_module/dev_utils/optimize_locate_params.py \
--task set_snli \
--config new_module/set_consistency_energy/params_set_snli_gradnorm.yaml \
--data_path new_module/data/set_nli/eval2/set_nli_eval2.jsonl \
--output_dir new_module/set_consistency_energy/results/nli/set_nli/46853 \
--device cuda \
--agg_method median