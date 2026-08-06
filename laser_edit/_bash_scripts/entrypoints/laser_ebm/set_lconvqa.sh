#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --job-name=laser_ebm_lconvqa
#SBATCH --output='laser_edit/_slurm_outs/laser_ebm_lconvqa_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit-pro6000

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/.cache
export LOGGING_LEVEL=INFO

# Method: LASER & EBM edit | Task: set-LConVQA
# Hyperparams from paper Table 14: (w_f,w_c)=(1,100000), N=8, m=1, l=1, k=5, n_b=5
# Localization from Table 13 / params_set_lconvqa.yaml: Attention avg + GradNorm (l=1)

srun python laser_edit/ebm_edit_main_sc_energy.py \
set_lconvqa laser_edit/data/lconvqa/testset_incon_300/lconvqa_testset_incon_300.jsonl \
--slurm_job_id $SLURM_JOB_ID \
--early_stopping_patience 0 \
--losses gpt2_no_prefix sc_energy \
--thresholds -1 \
--loss_weights 1 100000 \
--k_per_location 5 \
--beam_size 5 \
--n_iter 8 \
--max_tokens_per_span 1 \
--dont_skip_allsat \
--selection_criteria allsat_primary \
--wandb_project sc_energy \
--wandb_entity hayleyson \
--ebm_params_path laser_edit/set_consistency_energy/params_set_lconvqa.yaml \
--causal_lm_path Qwen/Qwen2.5-7B-Instruct \
--mlm_path roberta-base \
--locate_mode ebm \
--num_edit_tokens_per_step 1 \
--output_dir_prefix 'outputs/sc_energy/set_lconvqa/ebm/'
