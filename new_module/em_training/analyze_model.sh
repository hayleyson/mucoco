#!/bin/bash
#SBATCH --job-name=formality-ranker
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --partition=P1
#SBATCH --output='slurm_output/%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds/epoch_6\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds/epoch_6/analysis/valid
 
srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds/epoch_6\
 --test_data_path data/formality/PT16/train.tsv\
 --test_data_type train\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds/epoch_6/analysis/train

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-filtered/epoch_5\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-filtered/epoch_5/analysis/valid
 
srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-filtered/epoch_5\
 --test_data_path data/formality/PT16/train.tsv\
 --test_data_type train\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-filtered/epoch_5/analysis/train

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-filtered/epoch_5\
 --test_data_path data/formality/PT16/train_filtered.tsv\
 --test_data_type train_filtered\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-filtered/epoch_5/analysis/train_filtered

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered/epoch_6\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered/epoch_6/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered-val-loss-mse/epoch_5\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered-val-loss-mse/epoch_5/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered/epoch_5\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered/epoch_5/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered-val-loss-mse/epoch_1\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered-val-loss-mse/epoch_1/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered-val-loss-mse-no-margin/epoch_15\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered-val-loss-mse-no-margin/epoch_15/analysis/valid\
 --rescale_labels



srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-val-loss-mse-no-margin-1/epoch_2\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-val-loss-mse-no-margin-1/epoch_2/analysis/valid\
 --rescale_labels


srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-no-margin/epoch_7\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-no-margin/epoch_7/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-val-loss-mse-no-margin-2/epoch_8\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-val-loss-mse-no-margin-2/epoch_8/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-val-loss-mse-no-margin-3/epoch_5\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-val-loss-mse-no-margin-3/epoch_5/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss/epoch_3\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss/epoch_3/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-scaled-ranking-loss/epoch_2\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-scaled-ranking-loss/epoch_2/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.1/epoch_2\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.1/epoch_2/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.01/epoch_4\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.01/epoch_4/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.2/epoch_6\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.2/epoch_6/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.3/epoch_5\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.3/epoch_5/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.15/epoch_8\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.15/epoch_8/analysis/valid\
 --rescale_labels

srun python examples/training_constraint_models/analyze_model.py\
 --checkpoint_dir models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.025/epoch_6\
 --test_data_path data/formality/PT16/valid.tsv\
 --test_data_type valid\
 --output_dir examples/training_constraint_models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.025/epoch_6/analysis/valid\
 --rescale_labels
