#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --nodelist=n02
#SBATCH --output='new_module/_slurm_outs/postproc_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python new_module/evaluation/postprocess_generations.py \
--input_dir /home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/ebm/j18pi8ab/final \
--save_dir /home/hyeryung/data/mucoco/outputs/toxicity/gpt3_5_gen/ebm/j18pi8ab/fluency \
--suffix jsonl \
--option for_fluency_metric