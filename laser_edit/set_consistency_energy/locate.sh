#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --output='laser_edit/_slurm_outs/set_consistency_locate_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit


export OPENAI_API_KEY=
export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

export LOGGING_LEVEL=DEBUG

  
python /home/hyeryung/data/mucoco/laser_edit/set_consistency_energy/locate.py \
--config /home/hyeryung/data/mucoco/laser_edit/set_consistency_energy/params_set_lconvqa_subtraction.yaml \
--task vqa \
--dataset lconvqa \
--data_dir /home/hyeryung/data/mucoco/laser_edit/data/lconvqa \
--loss_type triplet \
--decomposition no