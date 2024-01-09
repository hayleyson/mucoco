#!/bin/bash
#SBATCH --time=0-72:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output='formality_decoding_%j.out'
#SBATCH --nodelist=n02

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python new_module/mlm_reranking_all.py --method mlm-beamsearch-v1 \
--num_edit_token_per_step 5  \
--k_per_location 10 \
--n_iter 3 \
--selection_criteria weighted_sum \
--closs_weight 0.434540902741646 \
--beam_size 3 \
--task formality \
--source_data 'data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' \
--source_style 'informal' \
--target_style 'formal' \
--target_label_ids 1 1 \
--model_paths 'gpt2-large' 'models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17' \
--tokenizer_paths 'gpt2-large' 'models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17' \
--output_dir_prefix 'outputs/formality/mlm-reranking/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale'