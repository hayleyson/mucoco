#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-20:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=llm_edit
#SBATCH --output='/data3/saeheeeom/set_consistency/mucoco/new_module/_slurm_outs/edited_gpt_%j.out'
#SBATCH --nodelist=n01

source /data3/saeheeeom/.bashrc
source /data3/saeheeeom/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data3/saeheeeom/.cache
export HF_DATASETS_CACHE=/data3/saeheeeom/.cache
export TRANSFORMERS_CACHE=/data3/saeheeeom/.cache
export LOGGING_LEVEL=INFO

JOB_ID=$SLURM_JOB_ID

srun python /data3/saeheeeom/set_consistency/mucoco/new_module/llm_experiments/edit_with_llm/edit/llm_generate_jsonl.py \
--hf_model_name google/gemma-2-2b-it \
--file_save_path /data3/saeheeeom/set_consistency/mucoco/new_module/llm_experiments/edit_with_llm/edit/edited/edited_gemma_2b_0shot_$JOB_ID.jsonl \
--input_file_path /home/saeheeeom/data/set_consistency/mucoco/new_module/locate/located_toxicity/tox_0shot_28897.jsonl \
--orig_text_path \
--num_return_sequences 1 \
--max_tokens 150 \
--prompt_type nontoxic_both
# --run_type gen_all 
#--prompt_type nontoxic_both
#--run_type test_prompts
#--run_type gen_all
#--run_type gen_all
# --max_tokens 500

# dev_set_neg_36184.jsonl
# dev_set_pos_36724.jsonl

# tox_output_0shot_28897
# tox_output_noprompt_28898

# gpt-3.5-turbo-0125
# Qwen/Qwen2.5-7B-Instruct
# google/gemma-2-9b-it

# meta-llama/Meta-Llama-3.1-8B-Instruct
# microsoft/Phi-3.5-mini-instruct
# mistralai/Mistral-7B-Instruct-v0.3