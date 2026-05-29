#!/bin/bash
#SBATCH --job-name=t_cb_em
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodelist=n02
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output='laser_edit/_slurm_outs/t_bv0_clsf_%j.out'

source /home/hyeryung/.bashrc
source /home/hyeryung/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache
export LOGGING_LEVEL=INFO

srun python laser_edit/ebm_edit_main.py --method mlm-reranking \
--num_edit_token_per_step 5  \
--locate_unit token \
--k_per_location 4 \
--n_iter 10 \
--loss_weights 0.1 0.9 \
--selection_criteria allsat_primary \
--task toxicity \
--num_samples 10 \
--source_data 'laser_edit/data/toxicity-avoidance/testset_gpt2_2500.jsonl' \
--source_style 'toxic' \
--target_style 'nontoxic' \
--target_label_ids 0 0 \
--thresholds 0.9 \
--wandb_project 'toxicity-decoding' \
--model_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--tokenizer_paths 'gpt2-large' '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint' \
--model_types "AutoModelForCausalLM" "AutoModelForSequenceClassification" \
--output_dir_prefix 'outputs/toxicity/final' \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--locate_method 'grad_norm' \
--server_time_limit 12
