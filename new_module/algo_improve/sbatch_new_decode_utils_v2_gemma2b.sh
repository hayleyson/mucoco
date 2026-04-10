#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20GB
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --job-name=decode_util_v2
#SBATCH --output='new_module/_slurm_outs/decode_util_v2_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/data/hyeryung
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

python /home/hyeryung/data/mucoco/new_module/new_decode_utils_v2_comparing_v0_v1.py --method 0 --num_test_samples 50 --fluency_em_path google/gemma-2-2b
python /home/hyeryung/data/mucoco/new_module/new_decode_utils_v2_comparing_v0_v1.py --method 1 --num_test_samples 50 --fluency_em_path google/gemma-2-2b
# python /home/hyeryung/data/mucoco/new_module/new_decode_utils_v2_final.py --method final_gemma