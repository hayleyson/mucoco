#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_llama_toxic
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


# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_edit_prompted_generation.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_editing_gpt2_gens_nontoxic.jsonl \
# --input_file_path /data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500_locate.jsonl \
# --prompt_type nontoxic_gpt2_gen_edit \
# --num_return_sequences 1 \
# --max_tokens 50

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_edit_prompted_generation.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_editing_gpt2_gens_all_nontoxic.jsonl \
# --input_file_path /data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl \
# --prompt_type nontoxic_gpt2_gen_edit_3shot \
# --num_return_sequences 1 \
# --max_tokens 50

srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_edit_prompted_generation.py \
--hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
--file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/positive/editing/llama3_8b_editing_gpt2_gens_pos.jsonl \
--input_file_path /data/hyeryung/mucoco/new_module/data/sentiment/dev_set.jsonl \
--prompt_type senti_pos_gpt2_gen_edit_3shot \
--num_return_sequences 1 \
--max_tokens 50

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_edit_prompted_generation.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/negative/editing/llama3_8b_editing_gpt2_gens_neg.jsonl \
# --input_file_path /data/hyeryung/mucoco/new_module/data/sentiment/dev_set.jsonl \
# --prompt_type senti_neg_gpt2_gen_edit_3shot \
# --num_return_sequences 1 \
# --max_tokens 50