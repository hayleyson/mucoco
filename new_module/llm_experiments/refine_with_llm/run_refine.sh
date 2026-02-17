#!/bin/bash
#SBATCH --nodelist=n01
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=refine_test
#SBATCH --output='new_module/_slurm_outs/refine_test_%A_%a.out'
#SBATCH --array=0

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

# Define input/output paths for each array task
INPUT_PATHS=(
    # "outputs/nli/ay8ohdbp/outputs_epsilon0.99.txt"
    "outputs/toxicity/llm/j18pi8ab/outputs_epsilon0.95.txt.0"
)

OUTPUT_PATHS=(
    # "outputs/nli/ay8ohdbp/outputs_epsilon0.99_qwen2.5_7B_refined.txt"
    "/data/hyeryung/mucoco/outputs/toxicity/llm/j18pi8ab/outputs_epsilon0.95_qwen2.5_7B_refined.txt.0"
)

srun python new_module/llm_experiments/refine_with_llm/refine_postprocessing.py \
--input_path ${INPUT_PATHS[$SLURM_ARRAY_TASK_ID]} \
--output_path ${OUTPUT_PATHS[$SLURM_ARRAY_TASK_ID]} \
--consider_prefix True



