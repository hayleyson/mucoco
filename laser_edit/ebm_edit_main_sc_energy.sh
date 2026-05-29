#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --job-name=sc_energy_le
#SBATCH --output='laser_edit/_slurm_outs/sc_energy_le_%j.out'


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache


export LOGGING_LEVEL=INFO

# # set_snli
# '/home/hyeryung/data/mucoco/laser_edit/data/set_nli/processed_data/set_nli_test1_edited_only.jsonl'
# laser_edit/set_consistency_energy/params_set_snli.yaml
# # set_lconvqa
# '/home/hyeryung/data/mucoco/laser_edit/data/convqa/processed_data/lconvqa_test1_edited_only.jsonl'
# laser_edit/set_consistency_energy/params_set_lconvqa.yaml

# srun python laser_edit/ebm_edit_main_sc_energy.py \
# set_lconvqa /home/hyeryung/data/mucoco/laser_edit/data/convqa/processed_data/lconvqa_test1_edited_only.jsonl \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --losses gpt2_no_prefix sc_energy \
# --thresholds -1 \
# --loss_weights 1 10 \
# --k_per_location 5 \
# --beam_size 5 \
# --n_iter 8 \
# --dont_skip_allsat \
# --selection_criteria allsat_primary \
# --wandb_project sc_energy \
# --wandb_entity hayleyson \
# --params_path laser_edit/set_consistency_energy/params_set_lconvqa.yaml


# srun python laser_edit/ebm_edit_main_sc_energy.py \
# set_snli /home/hyeryung/data/mucoco/laser_edit/data/set_nli/processed_data/set_nli_test1_edited_only.jsonl \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --losses gpt2_no_prefix sc_energy \
# --thresholds -1 \
# --loss_weights 1 10 \
# --k_per_location 5 \
# --beam_size 5 \
# --n_iter 8 \
# --dont_skip_allsat \
# --selection_criteria allsat_primary \
# --wandb_project sc_energy \
# --wandb_entity hayleyson \
# --params_path laser_edit/set_consistency_energy/params_set_snli.yaml

# srun python laser_edit/ebm_edit_main_sc_energy.py \
# set_lconvqa /home/hyeryung/data/mucoco/laser_edit/data/convqa/processed_data/lconvqa_test1_edited_only.jsonl \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --losses gpt2_no_prefix sc_energy \
# --thresholds -1 \
# --loss_weights 1 10 \
# --k_per_location 5 \
# --beam_size 5 \
# --n_iter 8 \
# --dont_skip_allsat \
# --selection_criteria allsat_primary \
# --wandb_project sc_energy \
# --wandb_entity hayleyson \
# --params_path laser_edit/set_consistency_energy/params_set_lconvqa_clsf.yaml

# srun python laser_edit/ebm_edit_main_sc_energy.py \
# set_snli /home/hyeryung/data/mucoco/laser_edit/data/set_nli/processed_data/set_nli_test1_edited_only.jsonl \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --losses gpt2_no_prefix sc_energy \
# --thresholds -1 \
# --loss_weights 1 10 \
# --k_per_location 5 \
# --beam_size 5 \
# --n_iter 8 \
# --dont_skip_allsat \
# --selection_criteria allsat_primary \
# --wandb_project sc_energy \
# --wandb_entity hayleyson \
# --params_path laser_edit/set_consistency_energy/params_set_snli_clsf.yaml

# srun python laser_edit/ebm_edit_main_sc_energy.py \
# set_lconvqa laser_edit/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --losses gpt2_no_prefix sc_energy \
# --thresholds -1 \
# --loss_weights 1 10 \
# --k_per_location 5 \
# --beam_size 5 \
# --n_iter 8 \
# --dont_skip_allsat \
# --selection_criteria allsat_primary \
# --wandb_project sc_energy \
# --wandb_entity hayleyson \
# --ebm_params_path laser_edit/set_consistency_energy/params_set_lconvqa.yaml \
# --causal_lm_path gpt2-large \
# --mlm_path roberta-base

# srun python laser_edit/ebm_edit_main_sc_energy.py \
# set_lconvqa laser_edit/data/lconvqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl \
# --slurm_job_id $SLURM_JOB_ID \
# --early_stopping_patience 0 \
# --losses gpt2_no_prefix sc_energy \
# --thresholds -1 \
# --loss_weights 1 100000 \
# --k_per_location 5 \
# --beam_size 5 \
# --n_iter 8 \
# --dont_skip_allsat \
# --selection_criteria allsat_primary \
# --wandb_project sc_energy \
# --wandb_entity hayleyson \
# --ebm_params_path laser_edit/set_consistency_energy/params_set_lconvqa_ground_truth.yaml \
# --causal_lm_path Qwen/Qwen2.5-7B-Instruct \
# --mlm_path roberta-base \
# --locate_mode ground_truth \
# --num_edit_tokens_per_step 1 \
# --max_tokens_per_span 1 \
# --output_dir_prefix 'outputs/sc_energy/set_lconvqa/ebm/'

srun python laser_edit/ebm_edit_main_sc_energy.py \
set_nli laser_edit/data/set_nli/testset_incon_300/set_nli_testset_incon_300.jsonl \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--losses gpt2_no_prefix sc_energy \
--thresholds -1 \
--loss_weights 1 0.000002 \
--k_per_location 8 \
--beam_size 5 \
--n_iter 8 \
--max_tokens_per_span 3 \
--dont_skip_allsat \
--selection_criteria allsat_primary \
--wandb_project sc_energy \
--wandb_entity hayleyson \
--ebm_params_path laser_edit/set_consistency_energy/params_set_nli.yaml \
--causal_lm_path Qwen/Qwen2.5-7B-Instruct \
--mlm_path roberta-base \
--locate_mode ebm \
--num_edit_tokens_per_step 10 \
--output_dir_prefix 'outputs/sc_energy/set_nli/ebm/'