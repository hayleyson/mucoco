#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --output='laser_edit/_slurm_outs/evaluate_ebm_locate_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

export LOGGING_LEVEL=DEBUG

results_dir="laser_edit/locate/ebm/results"
output_dir="laser_edit/locate/ebm/evaluation_results"

prediction_file="laser_edit/locate/ebm/results/inconsistentspans/energy_model_8pylct20_gradient_norm_max_num_tokens_7.jsonl"
original_file="laser_edit/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl"
output_file="laser_edit/locate/ebm/evaluation_results/inconsistent_spans_energy_model_8pylct20_gradient_norm_max_num_tokens_7_evaluation_results.csv"

# python laser_edit/locate/ebm/evaluation_results/evaluate_locedit_loc_results.py \
# --task inconsistent \
# --results_dir $results_dir \
# --output_dir $output_dir

python laser_edit/locate/ebm/evaluation_results/evaluate_locedit_loc_results.py \
--task inconsistent \
--prediction_file $prediction_file \
--original_file $original_file \
--output_file $output_file