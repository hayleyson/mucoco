#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=sl_multi_pp
#SBATCH --output='laser_edit/_slurm_outs/self_loc_multi_pp_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

# Method: Self-locate & LLM edit | Task: multi | Step: postprocess (single joint locate file)

result_file="/home/hyeryung/data/mucoco/laser_edit/locate/llm/nli_toxicity/results/Qwen2.5-7B-Instruct_locate_toxic_incon_5shot_type1_nli_toxicity_rewrite_hypothesis_toxic_1782998231.jsonl"
original_file="/home/hyeryung/data/mucoco/laser_edit/data/nli-toxicity/Qwen3-8B_rewrite_hypothesis_toxic_5shot_test_set_260506.jsonl"
output_dir="laser_edit/locate/llm/nli_toxicity/processed_results"
mkdir -p "$output_dir"

srun -n 1 -c 1 python laser_edit/locate/llm/nli_toxicity/postprocess_results.py \
--result_file "$result_file" \
--original_file "$original_file" \
--output_file "$output_dir/$(basename "${result_file%.jsonl}")_processed.jsonl" \
--task nli_toxicity_rewrite_hypothesis_toxic
