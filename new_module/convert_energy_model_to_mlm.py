#!/usr/bin/env python
# coding: utf-8




# standard libraries
import os
import sys
import json
import logging
from collections import namedtuple
sys.path.append("/home/s3/hyeryung/mucoco")
os.chdir("/home/s3/hyeryung/mucoco")

# installed packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
import wandb
import random

# custom libraries
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
from mucoco.utils import RobertaCustomForSequenceClassification, TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, OptimizerLE, get_epsilon, locate
from new_module.evaluate_wandb import evaluate
from new_module.utils import score_hypotheses, constrained_beam_search_v0
from new_module.robertacustom import RobertaCustomForMaskedLM





model_path='models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best'





tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='hf_cache', use_fast=True)
config = AutoConfig.from_pretrained(model_path, cache_dir='hf_cache')
model = RobertaCustomForMaskedLM.from_pretrained(model_path, config=config, cache_dir='hf_cache')





mlm_model = AutoModelForMaskedLM.from_pretrained('roberta-base', cache_dir='hf_cache')





for i in model.lm_head.decoder.parameters():
    print(i)





model.lm_head = mlm_model.lm_head





for i in model.lm_head.decoder.parameters():
    print(i)





out_path = 'models/roberta-base-jigsaw-toxicity-mlm-with-gpt2-large-embeds/checkpoint_best'





model.save_pretrained(out_path)





tokenizer.save_pretrained(out_path)

