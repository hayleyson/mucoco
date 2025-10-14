#!/bin/bash
#SBATCH --nodelist=n01
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=nli_energy
#SBATCH --output='new_module/_slurm_outs/nli_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

# min epsilon에 대한 ablation
srun python new_module/new_mlm_reranking_all_sc_energy.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 5 \
--n_iter 1 \
--loss_weights 1 1 \
--selection_criteria allsat_primary \
--cache_dir '/data/hyeryung/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--task nli \
--output_dir_prefix 'outputs/sc_energy/' \
--source_data '/data/hyeryung/mucoco/new_module/data/convqa/processed_data/lconvqa_test_for_locate_edit_masked.pickle' \
--source_style 'inconsistent' \
--target_style 'consistent' \
--target_label_ids 1 1 \
--min_epsilons 0.99 \
--wandb_project 'sc-energy-decoding' \
--model_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/set_consistency_energy/results/vqa/lconvqa/1225068/SetCon-roberta-no-triplet-False-fg_tot.pth' \
--tokenizer_paths 'Qwen/Qwen2.5-7B-Instruct' '/data/hyeryung/set_consistency_energy/results/vqa/lconvqa/1225068/SetCon-roberta-no-triplet-False-fg_tot.pth' \
--locate_method 'grad_norm' \
--losses gpt2_no_prefix sc_energy \
--model_types AutoModelForCausalLM lossNet \
--dont_skip_allsat
