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

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-ranker \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --filtering \
# --margin=0.16666666666666666 \
# --checkpoint_path=models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered-val-loss-mse \
# --max_save_num=1 \
# --val_loss_type=mse_loss

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-ranker \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --filtering \
# --margin=0.16666666666666666 \
# --checkpoint_path=models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered \
# --max_save_num=1 \
# --val_loss_type=margin_ranking_loss

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor-multiloss \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --filtering \
# --margin=0.16666666666666666 \
# --checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered \
# --max_save_num=1 \
# --val_loss_type=margin_ranking_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.05

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor-multiloss \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --filtering \
# --margin=0.16666666666666666 \
# --checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-filtered-val-loss-mse \
# --max_save_num=1 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.05

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor-multiloss \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --margin=0.0 \
# --checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-no-margin \
# --max_save_num=1 \
# --val_loss_type=margin_ranking_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.05

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor-multiloss \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --margin=0.0 \
# --checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-val-loss-mse-no-margin \
# --max_save_num=1 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.05

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --margin=0.0 \
# --checkpoint_path=models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale-2 \
# --max_save_num=1 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.00

# python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --margin=0.0 \
# --checkpoint_path=models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale-3 \
# --max_save_num=1 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.00

# srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-scaled-ranking-loss \
# --max_save_num=1 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=0.0 \
# --loss_weight_ranking=1.00 \
# --ranking_loss_type=scaled_ranking_loss

# srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
# --model=custom-roberta-base-regressor \
# --batch_size=16 \
# --num_epochs=20 \
# --max_lr=5e-5 \
# --weight_decay=0.01 \
# --checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss \
# --max_save_num=1 \
# --val_loss_type=mse_loss \
# --loss_weight_mse=1.0 \
# --loss_weight_ranking=0.05 \
# --ranking_loss_type=scaled_ranking_loss

srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
--model=custom-roberta-base-regressor \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.1 \
--max_save_num=1 \
--val_loss_type=mse_loss \
--loss_weight_mse=1.0 \
--loss_weight_ranking=0.1 \
--ranking_loss_type=scaled_ranking_loss

srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
--model=custom-roberta-base-regressor \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.15 \
--max_save_num=1 \
--val_loss_type=mse_loss \
--loss_weight_mse=1.0 \
--loss_weight_ranking=0.15 \
--ranking_loss_type=scaled_ranking_loss

srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
--model=custom-roberta-base-regressor \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.2 \
--max_save_num=1 \
--val_loss_type=mse_loss \
--loss_weight_mse=1.0 \
--loss_weight_ranking=0.2 \
--ranking_loss_type=scaled_ranking_loss

srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
--model=custom-roberta-base-regressor \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.3 \
--max_save_num=1 \
--val_loss_type=mse_loss \
--loss_weight_mse=1.0 \
--loss_weight_ranking=0.3 \
--ranking_loss_type=scaled_ranking_loss

srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
--model=custom-roberta-base-regressor \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.025 \
--max_save_num=1 \
--val_loss_type=mse_loss \
--loss_weight_mse=1.0 \
--loss_weight_ranking=0.025 \
--ranking_loss_type=scaled_ranking_loss

srun python examples/training_constraint_models/train_energy_model_formality_ranking.py \
--model=custom-roberta-base-regressor \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--checkpoint_path=models/roberta-base-pt16-formality-regressor-multiloss-with-gpt2-large-embeds-scaled-ranking-loss-weight-ranking-0.01 \
--max_save_num=1 \
--val_loss_type=mse_loss \
--loss_weight_mse=1.0 \
--loss_weight_ranking=0.01 \
--ranking_loss_type=scaled_ranking_loss