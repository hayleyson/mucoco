#!/bin/bash
#SBATCH --nodelist=n01
#SBATCH --time=0-48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=formal_energy
#SBATCH --output='new_module/_slurm_outs/formality_decoding_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export HF_DATASETS_CACHE=/home/hyeryung/data/hf_cache
export TRANSFORMERS_CACHE=/home/hyeryung/data/hf_cache

# srun python new_module/new_mlm_reranking_all_sweep.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.74 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification


# srun python new_module/new_mlm_reranking_all_sweep_n_iter.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 7  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 10 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.74 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 1  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.74 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 1  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.74 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# srun python new_module/new_mlm_reranking_all_sweep_n_iter.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 1  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 10 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.74 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# srun python new_module/new_mlm_reranking_all_sweep.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 1  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 7 \
# --k_per_location 15 \
# --n_iter 1 \
# --loss_weights 1 1 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.74 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# ## ablation 결과가 달라진 것 파악하기 위해서 과거와 동일한 hyperparameter로 실험 (모델은 slightly 다름. gpt2-large-embed-share에서 embed-share 안하는 것으로 바뀜)
# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 3 \
# --loss_weights 0.83276342312 0.167236576878629 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.75 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification


# ## ablation 결과가 달라진 것 파악하기 위해서 과거와 동일한 hyperparameter로 실험 (모델까지 과거와 동일하게)
# srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --max_tokens_per_span 3 \
# --locate_unit word \
# --beam_size 5 \
# --k_per_location 10 \
# --n_iter 3 \
# --loss_weights 0.83276342312 0.167236576878629 \
# --selection_criteria allsat_primary \
# --cache_dir '/home/hyeryung/data/hf_cache' \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --dont_skip_allsat \
# --task formality \
# --output_dir_prefix 'outputs/formality/formal/' \
# --source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.75 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training/step_560_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training/step_560_best_checkpoint/' \
# --locate_method 'grad_norm' \
# --losses gpt2 classification_no_prefix_logprobloss \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

## ablation 결과가 달라진 것 파악하기 위해서 과거 model을 현재 hyperparameter로 실험
srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 7  \
--max_tokens_per_span 3 \
--locate_unit word \
--beam_size 7 \
--k_per_location 15 \
--n_iter 1 \
--loss_weights 1 1 \
--selection_criteria allsat_primary \
--cache_dir '/home/hyeryung/data/hf_cache' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--dont_skip_allsat \
--task formality \
--output_dir_prefix 'outputs/formality/formal/' \
--source_data '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
--source_style 'informal' \
--target_style 'formal' \
--target_label_ids 1 1 \
--min_epsilons 0.74 \
--wandb_project 'formality-decoding' \
--model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training/step_560_best_checkpoint/' \
--tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-energy-training/step_560_best_checkpoint/' \
--locate_method 'grad_norm' \
--losses gpt2 classification_no_prefix_logprobloss \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification