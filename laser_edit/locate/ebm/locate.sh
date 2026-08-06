#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --output='laser_edit/_slurm_outs/loc_skiml_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache


srun -n 1 -c 1 python /home/hyeryung/data/mucoco/laser_edit/locate/ebm/locate_utils.py \
--pretrained_model_path "/home/hyeryung/data/loc_edit/models/nli/roberta_base_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/8pylct20/" \
--input_file "/home/hyeryung/data/mucoco/laser_edit/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl" \
--output_file "/home/hyeryung/data/mucoco/laser_edit/locate/ebm/results/inconsistentspans/energy_model_8pylct20_gradient_norm_max_num_tokens_7.jsonl" \
--task inconsistentspans \
--label_id 1 \
--max_num_tokens 7 \
--locate_method grad_norm


srun -n 1 -c 1 python /home/hyeryung/data/mucoco/laser_edit/locate/ebm/locate_utils.py \
--pretrained_model_path "/home/hyeryung/data/loc_edit/models/nli/roberta_base_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/8pylct20/" \
--input_file "/home/hyeryung/data/mucoco/laser_edit/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl" \
--output_file "/home/hyeryung/data/mucoco/laser_edit/locate/ebm/results/inconsistentspans/energy_model_8pylct20_attention_max_num_tokens_7.jsonl" \
--task inconsistentspans \
--label_id 1 \
--max_num_tokens 7 \
--locate_method attention

# srun -n 1 -c 1 python /home/hyeryung/data/mucoco/laser_edit/locate/ebm/locate_utils.py \
# --pretrained_model_path "/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint" \
# --input_file "/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl" \
# --output_file "/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332_gradient_norm_max_num_tokens_7.jsonl" \
# --task toxicity \
# --label_id 0 \
# --max_num_tokens 7 \
# --locate_method grad_norm

