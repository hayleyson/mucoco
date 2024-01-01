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

python examples/training_constraint_models/train_energy_model_formality_ranking.py \
--model=custom-roberta-base-ranker \
--batch_size=16 \
--num_epochs=20 \
--max_lr=5e-5 \
--weight_decay=0.01 \
--filtering \
--margin=0.16666666666666666 \
--checkpoint_path=models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered \
--max_save_num=1 \
--val_loss_type=margin_ranking_loss