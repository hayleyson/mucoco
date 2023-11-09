#!/bin/bash
#SBATCH --job-name=eval_only_mlm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --partition=P2
#SBATCH --output='slurm_output/%j.out'

source ~/.bashrc
source ~/conda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH="{$PYTHONPATH}:/home/s3/hyeryung/mucoco"
srun python /home/s3/hyeryung/mucoco/ell-e/evaluate-only-wandb.py --run_path hayleyson/mucola/noe1aj78 --outfile /home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/mlm-token-nps4-k3-allsat_primary-0.5-0.5-wandb-2/outputs_epsilon-3-test.txt
