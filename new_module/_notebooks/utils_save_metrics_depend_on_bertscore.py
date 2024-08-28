from glob import glob
import numpy as np
import os
os.chdir('/data/hyeryung/mucoco')
import pandas as pd
from evaluation.prompted_sampling.evaluate import distinctness, repetition
from transformers import AutoTokenizer

## TOXICITY TASK
## 0. original output의 toxicity, ppl, cola, repetitions 파일을 읽어옴
## run마다 다음 작업을 한다.
## 1. wandb 연결
## 2. toxicity, ppl, cola, repetitions, bertscore 파일을 읽어옴
## 3. sbert, sbert_preserved_prop, sbert_preserved_count가 없는 경우, 쓰기
##    - sbert가 없는 경우, original outputs와 현재 run의 outputs 파일 추가적으로 읽기 -> bertscore 계산 코드 호출 -> sbert 쓰기
##    - bertscore 파일 읽어와서 sbert_preserved_prop, sbert_preserved_count 계산. -> 쓰기 
## 4. 1에서 읽어온 값들을 같은 row별로 합침
## 5. bertscore가 0.5 이상인지 미만인지에 따라 metric을 다시 구함
## 6. (!!!) wandb에 해당 값을 씀
##    - toxic_proba_bert_geq_50,toxic_proba_bert_lt_50
##    - ppl_bert_geq_50,ppl_bert_lt_50
##    - cola_bert_geq_50,cola_bert_lt_50
##    - rep_bert_geq_50,rep_bert_lt_50
## 7. bertscore가 0.5 미만인 row에 해당하는 original output의 toxicity, ppl, cola, repetitions 값을 가져옴
## 8. 5에서 가져온 값과 bertscore가 0.5 이상인 row에 해당하는 toxicity, ppl, cola, repetitions 값을 concat함
## 9. concat한 값을 기준으로 metric을 다시 구함
## 10. (!!!) wandb에 해당 값을 씀
##    - toxic_proba_bg50_ro
##    - ppl_bg50_ro
##    - cola_bg50_ro
##    - rep_bg50_ro

wandb_path = ''
## 0. original output의 toxicity, ppl, cola, repetitions 파일을 읽어옴
## run마다 다음 작업을 한다.
## 1. wandb 연결
## 2. toxicity, ppl, cola, repetitions, bertscore 파일을 읽어옴
## 3. sbert, sbert_preserved_prop, sbert_preserved_count가 없는 경우, 쓰기
##    - sbert가 없는 경우, original outputs와 현재 run의 outputs 파일 추가적으로 읽기 -> bertscore 계산 코드 호출 -> sbert 쓰기
##    - bertscore 파일 읽어와서 sbert_preserved_prop, sbert_preserved_count 계산. -> 쓰기 
## 4. 1에서 읽어온 값들을 같은 row별로 합침
## 5. bertscore가 0.5 이상인지 미만인지에 따라 metric을 다시 구함
## 6. (!!!) wandb에 해당 값을 씀
##    - toxic_proba_bert_geq_50,toxic_proba_bert_lt_50
##    - ppl_bert_geq_50,ppl_bert_lt_50
##    - cola_bert_geq_50,cola_bert_lt_50
##    - rep_bert_geq_50,rep_bert_lt_50
## 7. bertscore가 0.5 미만인 row에 해당하는 original output의 toxicity, ppl, cola, repetitions 값을 가져옴
## 8. 5에서 가져온 값과 bertscore가 0.5 이상인 row에 해당하는 toxicity, ppl, cola, repetitions 값을 concat함
## 9. concat한 값을 기준으로 metric을 다시 구함
## 10. (!!!) wandb에 해당 값을 씀
##    - toxic_proba_bg50_ro
##    - ppl_bg50_ro
##    - cola_bg50_ro
##    - rep_bg50_ro

neg_file = '/data/hyeryung/mixmatch/output_samples/form_em_1/merged/opt_samples' ## informal 
pos_file = '/data/hyeryung/mixmatch/output_samples/form_em_0/disc_frm_new_data_form_em_test_sh8_len_b_sc_r_inf_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_300.0_date_16_04_2024_00_23_29/opt_samples' ## formal
tokenizer = AutoTokenizer.from_pretrained('gpt2-large')

data = []
for fpath in sorted(glob(f'{pos_file}.txt')+glob(f'{neg_file}.txt')):
    with open(fpath, 'r') as f:
        output = [[{'text': x.rstrip(),
                    'tokens': tokenizer.encode(x.rstrip(), add_special_tokens=False)}] for x in f.readlines()]
        output = pd.DataFrame({'generations': output})
        output['prompt'] = [{"text": "<|endoftext|>", "tokens": [50256]} for _ in range(output.shape[0])]

    data.append(output)
data = pd.concat(data, axis=0)

data =data.explode('generations')
all_data = []
for fpath in sorted(glob(f'{pos_file}*fluency')+glob(f'{neg_file}*fluency')):
    print(fpath)
    with open(fpath , 'r') as f:
        tmp_data = [1 if x.strip() == 'LABEL_1' else 0 for x in f.readlines()]
    all_data.extend(tmp_data)
data['fluency'] = all_data
print(np.mean(all_data))
all_data = []
# for fpath in sorted(glob('sentiment/sentiment/*.ppl-big')):
for fpath in sorted(glob(f'{pos_file}*ppl-big')+glob(f'{neg_file}*ppl-big')):
    with open(fpath , 'r') as f:
        tmp_data = [float(x.strip().split(',')[0]) for x in f.readlines()]
    all_data.extend(tmp_data)
data['ppl'] = all_data
print(np.mean(all_data))
all_data = []
# for fpath in sorted(glob('sentiment/sentiment/*.repetitions')):
for fpath in sorted(glob(f'{pos_file}*.repetitions')+glob(f'{neg_file}*.repetitions')):
    with open(fpath , 'r') as f:
        tmp_data = [0 if x.strip() == "{}" else 1 for x in f.readlines()]
    all_data.extend(tmp_data)
data['rep']=all_data
print(np.mean(all_data))

## dist-3

dist3_metrics=[]
for fpath in sorted(glob(f'{pos_file}.txt')+glob(f'{neg_file}.txt')):
    print(fpath)
    with open(fpath, 'r') as f:
        outputs = [[{'text': x.rstrip()}] for x in f.readlines()]
        outputs = pd.DataFrame({'generations': outputs})
        outputs['prompt'] = [{"text": "<|endoftext|>", "tokens": [50256]} for _ in range(outputs.shape[0])]

    _,_,dist3=distinctness(outputs)
    dist3_metrics.append(dist3)

print(np.mean(dist3_metrics))
all_data = []
# for fpath in sorted(glob('sentiment/sentiment/*.sbertscore')):
for fpath in sorted(glob(f'{pos_file}*.sbertscore')+glob(f'{neg_file}*.sbertscore')):
    with open(fpath , 'r') as f:
        raw_data = f.readlines()
        tmp_data = [float(x.strip()) for x in raw_data[1:]]
    all_data.extend(tmp_data)
data['bert']=all_data
print(np.mean(all_data))

all_data = []
for fpath in sorted(glob(f'{pos_file}*formality_ext')):
    
    with open(fpath , 'r') as f:
        raw_data = f.readlines()
        tmp_data = [1 if float(x) >= 0.5 else 0 for x in raw_data]
    
    all_data.extend(tmp_data)
print(np.mean(tmp_data))
    
for fpath in sorted(glob(f'{neg_file}*formality_ext')):
    
    with open(fpath , 'r') as f:
        raw_data = f.readlines()
        tmp_data = [1 if float(x) < 0.5 else 0 for x in raw_data]
    all_data.extend(tmp_data)
    
data['formality_acc'] = all_data
print(np.mean(tmp_data))
print(np.mean(all_data))

data['sbert_geq_50']=data['bert'] >= 0.5
data.groupby('sbert_geq_50')[['formality_acc', 'ppl', 'fluency','rep', 'bert',
       ]].mean().sort_index(ascending=False)
data.groupby('sbert_geq_50').size().sort_index(ascending=False)
data.groupby('sbert_geq_50').size().sort_index(ascending=False)/data.shape[0]
data['seq_len'] = data['generations'].apply(lambda x: len(x['tokens']))
data.seq_len.sum()