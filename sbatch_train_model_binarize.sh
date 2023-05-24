#!/bin/bash
#SBATCH --job-name=energy-model
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=8
#SBATCH --output='/home/hyeryungson/mucoco/slurm_output/%j.out'

start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

source /home/${USER}/.bashrc
source ~/anaconda/etc/profile.d/conda.sh
conda activate mucoco

export LOG_LEVEL=DEBUG
srun python -m notebooks.energy_model_retrain.allow_infinite_run_binarize \
--resume $1 \
--epochs 10 \
--wandb_name 'label-binarized-balanced' \
# --run_id 'sa094i7f'
# @click.option('--resume', default=False, help='Whether to resume previously stopped run')
# @click.option('--epochs', default=10, help='Total number of training epochs')
# @click.option('--warmup_steps', default=600, help='Number of steps for learning rate warm up')
# @click.option('--learning_rate', default=1e-5, help='Initial learning rate')
# @click.option('--weight_decay', default=0.01, help='Strength of weight decay for AdamW')
# @click.option('--logging_steps', default=100, help='Number of steps for which to log')
# @click.option('--eval_steps', default=500, help='Number of steps for which to evaluate')
# @click.option('--evaluation_strategy', default="steps", help='Evaluation strategy')
# @click.option('--save_total_limit', default=5, help='Number of checkpoints to keep track of.')
# @click.option('--per_device_train_batch_size', default=4, help='Batch size per device during training')
# @click.option('--per_device_eval_batch_size', default=4, help='Batch size for evaluation')
# @click.option('--gradient_accumulation_steps', default=4, help='Steps to accumulate gradients')
# @click.option('--metric_for_best_model', default="eval_loss", help='Metric to decide best model after training completes')
# @click.option('--greater_is_better', default=False, help='Whether metric_for_best_model is the better the greater')

