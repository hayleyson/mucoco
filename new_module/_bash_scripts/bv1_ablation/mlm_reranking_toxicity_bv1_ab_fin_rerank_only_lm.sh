#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --job-name=t_bv1_ab
#SBATCH --output='new_module/_slurm_outs/t_bv1_ab_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO


# ~selection_criteria 를 weighted sum으로 해주어야 함.~ -> 코드를 수정하고 나서는 꼭 그럴 필요가 없음. allsat으로 실험
# history: 원래는 LM 만 고려해서 update 여부까지 결정했는데, 그렇게 하니까 아예 update 자체가 안되어서 
# update 여부 결정 시에는 EM 과 LM 을 모두 고려하도록 코드를 수정. (수정방법: final_reranking에서 weighted sum도 제대로 계산하고 다만 그 함수 안에서의 best 결정 때만 fluency score만 고려하도록 수정)
srun python new_module/new_mlm_reranking_all_bv1_ab.py \
--method mlm-beamsearch-v1 \
--num_edit_token_per_step 5 \
--locate_unit word \
--k_per_location 10 \
--n_iter 10 \
--loss_weights 0.1 0.9 \
--beam_size 3 \
--selection_criteria allsat_primary \
--task toxicity \
--num_samples 10 \
--source_data new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl \
--source_style toxic \
--target_style nontoxic \
--target_label_ids 0 0 \
--min_epsilons 0.9 \
--wandb_project toxicity-decoding \
--model_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
--tokenizer_paths gpt2-large /data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint \
--output_dir_prefix outputs/toxicity/final \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method grad_norm \
--server_time_limit 24 \
--device cuda \
--dont_skip_allsat \
--cache_dir /data/hyeryung/hf_cache \
--model_types AutoModelForCausalLM AutoModelForSequenceClassification