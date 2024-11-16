#!/bin/bash
#SBATCH --job-name=t_bv0_clsf
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --nodelist=n01
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/t_bv0_clsf_%j.out'

source /home/hyeryung/.bashrc
source /home/hyeryung/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache
export LOGGING_LEVEL=INFO

# ## 일단은 skip 허용해서 실험 돌렸음 
# ## toxicity
# ## 4kp4ti6s 의 설정을 배끼되, loss_weights만 0.1 1.0으로 바꿈 
# ## 2트. min_epsilons를 0.75로도 수행
# srun python new_module/new_mlm_reranking_all.py \
# --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5 \
# --locate_unit word \
# --beam_size 3 \
# --k_per_location 10 \
# --n_iter 10 \
# --loss_weights 0.1 1.0 \
# --selection_criteria allsat_primary \
# --task toxicity \
# --num_samples 10 \
# --source_data new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl \
# --source_style toxic \
# --target_style nontoxic \
# --target_label_ids 0 0 \
# --min_epsilons 0.75 \
# --wandb_project toxicity-decoding \
# --model_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
# --tokenizer_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
# --output_dir_prefix outputs/toxicity/final \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method grad_norm \
# --server_time_limit 48 \
# --device cuda \
# --consider_prompt_for_cand_gen \
# --cache_dir /data/hyeryung/hf_cache \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification

# ## 일단은 skip 허용해서 실험 돌렸음 
# # sentiment positive
# # 2xn81iv5 의 설정에서 loss weigths 0.1 1.0 으로 하고 min_epsilons 1.0으로 보내버렸음 
# # update 24/10/05: loss weight에 대해 sweep 할 수 있도록 설정
# srun python new_module/new_mlm_reranking_all_sweep.py \
# --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5 \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 10 \
# --beam_size 3 \
# --loss_weights 0.1 1.0 \
# --selection_criteria allsat_primary \
# --task sentiment \
# --num_samples 20 \
# --source_data new_module/data/sentiment/dev_set.jsonl \
# --source_style negative \
# --target_style positive \
# --target_label_ids 1 1 \
# --min_epsilons 1.0 \
# --wandb_project sentiment-decoding \
# --model_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint \
# --tokenizer_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification \
# --output_dir_prefix outputs/sentiment/final \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method grad_norm \
# --server_time_limit 48 \
# --device 'cuda' \
# --cache_dir '/data/hyeryung/hf_cache' \
# --consider_prompt_for_cand_gen \
# --sweep

## 일단은 skip 허용해서 실험 돌렸음 
# sentiment negative
# 위 설정에서 target_labels_id만 0 0 으로 변경
# update 24/10/05: loss weight에 대해 sweep 할 수 있도록 설정
srun python new_module/new_mlm_reranking_all_sweep_.py \
--method mlm-beamsearch-v0 \
--num_edit_token_per_step 5 \
--locate_unit word \
--k_per_location 10 \
--n_iter 10 \
--beam_size 3 \
--loss_weights 0.1 1.0 \
--selection_criteria allsat_primary \
--task sentiment \
--num_samples 20 \
--source_data new_module/data/sentiment/dev_set.jsonl \
--source_style positive \
--target_style negative \
--target_label_ids 0 0 \
--min_epsilons 1.0 \
--wandb_project sentiment-decoding \
--model_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint \
--tokenizer_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification \
--output_dir_prefix outputs/sentiment/final \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method grad_norm \
--server_time_limit 48 \
--device 'cuda' \
--cache_dir '/data/hyeryung/hf_cache' \
--consider_prompt_for_cand_gen \
--sweep

# # formality
# # skip 허용 x 
# # cutgmg96 의 설정에서 loss_weight 0.1 1.0으로 업데이트 
# srun python new_module/new_mlm_reranking_all.py \
# --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5 \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 10 \
# --loss_weights 0.1 1.0 \
# --selection_criteria allsat_primary \
# --task formality \
# --source_data data/formality/GYAFC_Corpus/Entertainment_Music/test/informal \
# --source_style informal \
# --target_style formal \
# --target_label_ids 1 1 \
# --min_epsilons 0.9 \
# --wandb_project formality-decoding \
# --model_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/ \
# --tokenizer_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-pt16-formality-classifier-energy-training/step_1120_best_checkpoint/ \
# --model_types AutoModelForCausalLM AutoModelForSequenceClassification \
# --output_dir_prefix outputs/formality/final/ \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --locate_method grad_norm \
# --server_time_limit 48 \
# --dont_skip_allsat \
# --consider_prompt_for_cand_gen \
# --wandb_run_id 5dufnv1f \
# --resume