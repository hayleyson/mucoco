from pathlib import Path
from tqdm import tqdm 
import datasets 
import re
import json
import numpy as np
import pandas as pd
from IPython.display import display, HTML, Markdown
from pprint import pprint
from typing import Dict, List, Any, Optional
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score
import os, sys, json, math, glob, pickle, datetime, yaml
from pathlib import Path
import torch


import string
import os
import random
from typing import List, Tuple
from copy import deepcopy
from itertools import repeat
import logging

import pandas as pd
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

from laser_edit.ebm_training.nli.models.encoder import EncoderModel


def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]
    
file_paths = {
    'phase1_train': 'laser_edit/data/prm800k/phase1_train.jsonl', 
    'phase1_test': 'laser_edit/data/prm800k/phase1_test.jsonl', 
    'phase2_train': 'laser_edit/data/prm800k/phase2_train.jsonl', 
    'phase2_test': 'laser_edit/data/prm800k/phase2_test.jsonl',      
}

test_datasets = {
    'phase1_test': read_jsonl(file_paths['phase1_test']), 
    'phase2_test': read_jsonl(file_paths['phase2_test']), 
    'phase1_train': read_jsonl(file_paths['phase1_train']),
    'phase2_train': read_jsonl(file_paths['phase2_train']),
}    

# 1) 경로/환경 설정


# 프로젝트 루트로 sys.path 추가 (이 노트북은 repo 루트의 notebooks/ 하위에 있다고 가정)
# repo_root = Path('.').resolve()
repo_root = Path("laser_edit/set-consistency-demo")
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# CKPT_PATH = str((repo_root / 'checkpoint' / '86462' / 'SetCon-roberta-no-margin-False-fg_tot.pth').resolve())  # 필요시 변경
CKPT_PATH = str((repo_root / 'checkpoint' / '86316' / 'SetCon-longformer-no-margin-False-fg_tot.pth').resolve())  # 필요시 변경
CONFIG_YAML = str(repo_root / 'config.yaml')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# 2) 파라미터 로드/수정 (dataset=prm800k, repre_model=longformer, margin 등)


with open(CONFIG_YAML, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# 실험 환경 고정
params['dataset'] = 'prm800k'
params['task'] = 'prm'  # dataset_loader에서 KeyError 방지용 (분기엔 영향 없음)
params['pairwise'] = 'arbitrary_pairs'
params['time_key'] = ''
params['batch_size'] = 1
params['eval']['batch_size'] = 1
params['device'] = device

# 에너지넷 설정: Longformer + no decomposition + margin
# params['energynet']['repre_model'] = 'roberta'
params['energynet']['repre_model'] = 'longformer'
params['energynet']['decomposition'] = 'no'
params['energynet']['loss_type'] = 'margin'
params['energynet']['output_form'] = 'real_num'

# PRM800K 데이터셋 크기 (임계값 학습 시 사용)
params.setdefault('prm800k', {})
# 데이터 경로는 리포지토리 루트 기준으로 고정
params['prm800k']['data_dir'] = str((repo_root / 'datasets' / 'prm800k').resolve())
params['prm800k'].setdefault('stepwise_dataset_eval_num', 500)
params['prm800k'].setdefault('stepwise_dataset_test_num', 500)

# 체크포인트 경로 저장
params['model_path'] = CKPT_PATH
params['folder_path'] = str(Path(CKPT_PATH).parent)

# Locate 방식 관련 설정
params['locate']['type'] = 'gradnorm'
params['locate']['attentions_num_layer'] = 7
params['locate']['agg_method'] = 'avg'
params['locate']['select_method'] = 'max'

params

# 3) 모델 구성 및 체크포인트 로드
from energynets.energynet import energynet
import torch.nn as nn

def _clean_state_dict(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'): ]
        if nk.startswith('representation_model.module.'):
            nk = nk.replace('representation_model.module.', 'representation_model.')
        if nk.startswith('energy_model.representation_model.'):
            nk = nk.replace('energy_model.representation_model.', 'representation_model.')
        if nk.startswith('loss_function.decomposition.representation_model.'):
            nk = nk.replace('loss_function.decomposition.representation_model.', 'representation_model.')
        cleaned[nk] = v
    return cleaned

def _extract_rep_state(sd_in: dict) -> dict:
    rep_map = {}
    for k, v in sd_in.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'): ]
        anchor = 'representation_model.'
        idx = nk.find(anchor)
        if idx == -1:
            continue
        sub = nk[idx + len(anchor): ]
        if sub.startswith('module.'):
            sub = sub[len('module.'): ]
        rep_map[sub] = v
    return rep_map

lossNet = energynet(params).to(device)
state = torch.load(CKPT_PATH, map_location=device)
sd_raw = state.get('state_dict', state) if isinstance(state, dict) else state

# 1) 표현모델 가중치 우선 로드
try:
    rep_sd = _extract_rep_state(sd_raw)
    if rep_sd:
        target_rm = lossNet.representation_model
        if isinstance(target_rm, nn.DataParallel):
            target_rm = target_rm.module
        _res = target_rm.load_state_dict(rep_sd, strict=False)
        print('rep weights loaded')
except Exception as e:
    print('[WARN] rep load failed:', e)

# 2) 나머지 가중치 로드 (rep 제외)
try:
    sd_clean = _clean_state_dict(sd_raw)
    others = {k:v for k,v in sd_clean.items() if not k.startswith('representation_model.')}
    _missing, _unexpected = lossNet.load_state_dict(others, strict=False)
    print('other weights loaded (non-strict)')
except Exception as e:
    print('[WARN] fallback whole state (non-strict):', e)
    try:
        _ = lossNet.load_state_dict(sd_clean, strict=False)
    except Exception:
        _ = lossNet.load_state_dict(sd_raw, strict=False)

# threshold 복원 시도
if isinstance(state, dict) and ('threshold' in state):
    try:
        lossNet.threshold = float(state['threshold'])
        print('threshold from ckpt:', lossNet.threshold)
    except Exception:
        pass

lossNet.output_form, lossNet.threshold

# 5) 유틸: 세트 문자열 구성 및 에너지/판별 계산
from typing import List, Tuple
import torch

def build_set_text(question: str, steps: List[str], cls_token: str, sep_token: str) -> str:
    tmp = f'{cls_token} ' if cls_token else ''
    q = (question or '').strip()
    if q:
        tmp += q
        if sep_token:
            tmp += f' {sep_token} '
        else:
            tmp += ' '
    for i, s in enumerate(steps):
        s = (s or '').strip()
        if i != 0:
            tmp += ' '
        tmp += s
        if all([not tmp.endswith(mark) for mark in ['.', '!', '?']]):
            tmp += '.'
        if tmp.endswith('..'):
            tmp = tmp[:-1]
    return tmp

# how to figure out gold label


# how to figure out gold label


gold_labels_locate = []
gold_labels_cls = []
for item in test_datasets['phase2_train']:
    if item['label']['finish_reason'] == 'give_up':
        continue
    
    num_annotated_steps = len(item['label']['steps'])
    if num_annotated_steps == 0:
        continue
    
    if item['label']['finish_reason'] == 'solution': # correct CoT 
        gold_labels_locate.append(-1) # -1 for no wrong step
        gold_labels_cls.append(0)
    else: # 'found_error' # incorrect CoT
        if item['label']['steps'][num_annotated_steps - 1]['completions'][0]['rating'] != -1:
            raise ValueError(f"Gold label for the final step is not -1: {item['label']['steps'][num_annotated_steps - 1]}")
        gold_labels_locate.append(num_annotated_steps - 1) # final annotated step
        gold_labels_cls.append(1)
        

rm = lossNet.representation_model
cls_tok = getattr(rm.tokenizer, 'cls_token', '<s>')
sep_tok = getattr(rm.tokenizer, 'sep_token', '</s>')
print(f"cls_token: {cls_tok}, sep_token: {sep_tok}")

item_indexes = []
pred_labels = []

for i, item in tqdm(enumerate(test_datasets['phase2_train']), total=len(test_datasets['phase2_train'])):
    if item['label']['finish_reason'] == 'give_up':
        continue
    num_annotated_steps = len(item['label']['steps'])
    if num_annotated_steps == 0:
        continue
    
    
    item_indexes.append(i)
    question = item['question']['problem']
    steps = item['question'].get('pre_generated_steps', [])
    
    # 1) 전체 풀이 세트로 한 번 평가
    full_text = build_set_text(question, steps, cls_tok, sep_tok)
    e_val, _ = lossNet.energy_model([full_text], pair_only=True)
    # shape: (1,1) -> scalar
    if hasattr(e_val, 'shape') and e_val.shape[-1] == 1:
        e_full = float(e_val.view(-1)[0].item())
    else:
        e_full = float(e_val.view(-1)[0].item())
    if e_full < lossNet.threshold:
        pred_labels.append(0)
    else:
        pred_labels.append(1)
    # pretty_print_result(problem, solution, energies, decisions)
    
print(f"accuracy: {accuracy_score(gold_labels_cls, pred_labels)}")
    