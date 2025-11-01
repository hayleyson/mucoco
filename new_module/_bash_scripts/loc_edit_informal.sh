#!/bin/bash
#SBATCH -J Serial_gpu_job
#SBATCH -p gpu-farm
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=formal_energy
#SBATCH --output='new_module/_slurm_outs/formality_decoding_%j.out'

module purge module load cuda/12.1
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 5 \
--k_per_location 10 \
--n_iter 1 \
--loss_weights 0.1 1.0 \
--selection_criteria allsat_primary \
--cache_dir '/data/hyeryung/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task formality \
--output_dir_prefix 'outputs/formality/formal/' \
--source_data '/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
--source_style 'formal' \
--target_style 'informal' \
--target_label_ids 0 0 \
--min_epsilons 0.88 \
--wandb_project 'formality-decoding' \
--model_paths 'microsoft/Phi-3.5-mini-instruct' '/data/hyeryung/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
--tokenizer_paths 'microsoft/Phi-3.5-mini-instruct' '/data/hyeryung/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
--locate_method 'grad_norm' \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification

