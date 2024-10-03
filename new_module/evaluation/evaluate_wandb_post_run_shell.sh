#!/bin/bash
export PYTHONPATH=.
export HF_HOME=/data/hyeryung/hf_cache
export HF_DATASETS_CACHE=/data/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/data/hyeryung/hf_cache

python new_module/evaluation/evaluate_wandb_post_run_file_based.py