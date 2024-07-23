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
export CUDA_VISIBLE_DEVICES="2,3"

# srun python /data/hyeryung/mucoco/llm_experiments/llama_generate.py --hf_model_name meta-llama/Llama-2-13b-chat-hf --file_save_path /data/hyeryung/mucoco/llm_experiments/llama2_13b_chat_gens.jsonl
# srun python /data/hyeryung/mucoco/llm_experiments/llama_generate.py --hf_model_name meta-llama/Meta-Llama-3-8B --file_save_path /data/hyeryung/mucoco/llm_experiments/llama3_8b_gens.jsonl
# srun python /data/hyeryung/mucoco/llm_experiments/llama_generate.py --hf_model_name meta-llama/Meta-Llama-3-8B-Instruct --file_save_path /data/hyeryung/mucoco/llm_experiments/llama3_8b_instruct_gens.jsonl
srun python /data/hyeryung/mucoco/llm_experiments/llama_generate.py --hf_model_name meta-llama/Llama-2-70b-chat-hf --file_save_path /data/hyeryung/mucoco/llm_experiments/llama2_70b_chat_gens.jsonl