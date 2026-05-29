#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=n04
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=postprocess_results      
#SBATCH --output='laser_edit/_slurm_outs/postprocess_results_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Process all files in the results directory
results_dir="laser_edit/locate/llm/nli_toxicity/results"
original_toxic_file="laser_edit/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl"
original_inconsistent_file="laser_edit/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl"
output_dir="laser_edit/locate/llm/nli_toxicity/processed_results"

python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
--results_dir $results_dir \
--original_toxic_file $original_toxic_file \
--original_inconsistent_file $original_inconsistent_file \
--output_dir $output_dir