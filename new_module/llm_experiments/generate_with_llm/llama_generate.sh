#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_llama_toxic
#SBATCH --output='new_module/####_%j.out'
#SBATCH --nodelist=n01

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_generate.py \
--hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
--file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_chat_prompting_gens_informal_3shot_ungrammar__time_check.jsonl \
--input_file_path /data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal \
--prompt_type informal_3shot_ungrammar \
--num_return_sequences 1 \
--max_tokens 60

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_generate.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_chat_prompting_gens_informal_0shot_ungrammar.jsonl \
# --input_file_path /data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal \
# --prompt_type informal_0shot_ungrammar \
# --num_return_sequences 1 \
# --max_tokens 60

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_generate.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_chat_prompting_gens_formal_0shot.jsonl \
# --input_file_path /data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal \
# --prompt_type formal_0shot \
# --num_return_sequences 1 \
# --max_tokens 60

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_generate.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_chat_prompting_gens_formal_3shot__time_check.jsonl \
# --input_file_path /data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal \
# --prompt_type formal_3shot \
# --num_return_sequences 1 \
# --max_tokens 60

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_generate.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_chat_prompting_gens_nontoxic_3shot.jsonl \
# --input_file_path /data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl \
# --prompt_type nontoxic_3shot \
# --num_return_sequences 10 \
# --max_tokens 30

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_generate.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_chat_prompting_gens_nontoxic_0shot.jsonl \
# --input_file_path /data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl \
# --prompt_type nontoxic_0shot \
# --num_return_sequences 10 \
# --max_tokens 30

# srun python /data/hyeryung/mucoco/new_module/llm_experiments/llama_generate.py \
# --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --file_save_path /data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_chat_prompting_gens_nontoxic_0shot.jsonl \
# --input_file_path /data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl \
# --prompt_type nontoxic_0shot \
# --num_return_sequences 20 \
# --max_tokens 12