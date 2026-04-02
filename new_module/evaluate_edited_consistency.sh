#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --nodelist=n02
#SBATCH --output='new_module/_slurm_outs/evaluate_edit_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export OPENAI_API_KEY=

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export HF_DATASETS_CACHE=/home/hyeryung/data/hf_cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python new_module/evaluate_edited_consistency.py \
--edit_result_path outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen3-8b_w_ebm_locate_edit_result.jsonl \
--config_path new_module/set_consistency_energy/params_set_lconvqa.yaml \
--output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
--eval_model_id gpt-5-mini

# srun -n 1 -c 1 python new_module/evaluate_edited_consistency.py \
# --edit_result_path outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen3-8b_w_gt_locate_edit_result.jsonl \
# --config_path new_module/set_consistency_energy/params_set_lconvqa.yaml \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --eval_model_id gpt-5-mini

# srun -n 1 -c 1 python new_module/evaluate_edited_consistency.py \
# --edit_result_path outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen3-8b_w_self_locate_edit_result.jsonl \
# --config_path new_module/set_consistency_energy/params_set_lconvqa.yaml \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --eval_model_id gpt-5-mini

# srun -n 1 -c 1 python new_module/evaluate_edited_consistency.py \
# --edit_result_path outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen3-8b_wo_locate_edit_result.jsonl \
# --config_path new_module/set_consistency_energy/params_set_lconvqa.yaml \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --eval_model_id gpt-5-mini

# srun -n 1 -c 1 python new_module/evaluate_edited_consistency.py \
# --edit_result_path outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_gpt-5.4_none_w_gt_locate_edit_result.jsonl \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --eval_model_id gpt-5-mini

# srun -n 1 -c 1 python new_module/evaluate_edited_consistency.py \
# --edit_result_path outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_gpt-5.4_none_w_gt_locate_edit_result.jsonl \
# --output_dir outputs/sc_energy/set_lconvqa/llm/testset_incon_300/ \
# --eval_model_id gpt-5-mini