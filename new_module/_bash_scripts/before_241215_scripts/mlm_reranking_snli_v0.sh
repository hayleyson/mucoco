#!/bin/bash
#SBATCH --job-name=f_bv0_clsf
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --output='new_module/_slurm_outs/f_bv0_clsf_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

srun python new_module/new_mlm_reranking_all.py --method mlm-beamsearch-v0 \
--num_edit_token_per_step 5  \
--locate_unit word \
--k_per_location 10 \
--n_iter 3 \
--loss_weights 0.1 1.0 \
--selection_criteria allsat_primary \
--task nli \
--source_data '/data/hyeryung/mucoco/data/nli/snli_1.0/snli_1.0_test.jsonl' \
--source_style 'contradict' \
--target_style 'entail' \
--target_label_ids 0 0 \
--min_epsilons 0.75 \
--wandb_project 'nli-loc-edit' \
--model_paths 'gpt2-large' 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli' \
--tokenizer_paths 'gpt2-large' 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli' \
--output_dir_prefix 'outputs/nli/' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm'

# srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v0 \
# --num_edit_token_per_step 5  \
# --locate_unit word \
# --k_per_location 10 \
# --n_iter 3 \
# --closs_weight 0.167236576878629 \
# --selection_criteria allsat_primary \
# --task formality \
# --source_data 'data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
# --source_style 'informal' \
# --target_style 'formal' \
# --target_label_ids 1 1 \
# --min_epsilons 0.75 \
# --wandb_project 'formality-decoding' \
# --model_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/step_22500_best_checkpoint/' \
# --tokenizer_paths 'gpt2-large' '/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/step_22500_best_checkpoint/' \
# --output_dir_prefix 'outputs/formality/mlm-reranking/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds-gyafc/'