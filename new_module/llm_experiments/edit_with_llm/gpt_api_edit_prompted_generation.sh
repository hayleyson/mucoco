#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=gen_gpt4_toxic
#SBATCH --output='new_module/####_%j.out'

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun python /data/hyeryung/mucoco/new_module/llm_experiments/gpt_api_edit_prompted_generation.py \
--openai_api_key $OPENAI_API_KEY \
--file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_editing_gpt2_gens_nontoxic.jsonl \
--input_file_path /data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500_locate.jsonl \
--prompt_type nontoxic_gpt2_gen_edit \
--max_tokens 50 \
--num_test_prompts -1 \
--num_return_sequences 1