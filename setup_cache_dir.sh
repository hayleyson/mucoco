#!/bin/bash
## USAGE: bash setup_cache_dir.sh loc-edit2 /shared/s3/lab07/hyeryung/hf_cache
ENV_NAME=$1 #'loc-edit2'
CACHE_DIR=$2 #'/shared/s3/lab07/hyeryung/hf_cache'
conda env config vars set HF_HOME=$CACHE_DIR -n $ENV_NAME
conda env config vars set HF_DATASETS_CACHE=$CACHE_DIR -n $ENV_NAME
conda env config vars set TRANSFORMERS_CACHE=$CACHE_DIR -n $ENV_NAME