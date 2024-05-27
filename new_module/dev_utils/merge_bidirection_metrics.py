from glob import glob
import numpy as np
import os
os.chdir('/data/hyeryung/mucoco')
import pandas as pd
from evaluation.prompted_sampling.evaluate import distinctness, repetition

pos_file = 'outputs/formality/final/cutgmg96'
neg_file = 'outputs/formality/final/pe45pmd4_lowercase'

data = []
for fpath in sorted(glob(f'{pos_file}/outputs_epsilon*.txt')+glob(f'{neg_file}/outputs_epsilon*.txt')):
    output = pd.read_json(fpath, lines=True)
    output['fpath'] = fpath
    data.append(output)
data = pd.concat(data, axis=0)
data =data.explode('generations')

all_data = []
for fpath in sorted(glob(f'{pos_file}/*.fluency')+glob(f'{neg_file}/*.fluency')):
    with open(fpath , 'r') as f:
        tmp_data = [1 if x.strip() == 'LABEL_1' else 0 for x in f.readlines()]
    all_data.extend(tmp_data)
data['fluency'] = all_data
print(np.mean(all_data))

all_data = []
# for fpath in sorted(glob('sentiment/sentiment/*.ppl-big')):
for fpath in sorted(glob(f'{pos_file}/*.ppl-big')+glob(f'{neg_file}/*.ppl-big')):
    with open(fpath , 'r') as f:
        tmp_data = [float(x.strip().split(',')[0]) for x in f.readlines()]
    all_data.extend(tmp_data)
data['ppl'] = all_data
print(np.mean(all_data))

all_data = []
# for fpath in sorted(glob('sentiment/sentiment/*.repetitions')):
for fpath in sorted(glob(f'{pos_file}/*.repetitions')+glob(f'{neg_file}/*.repetitions')):
    with open(fpath , 'r') as f:
        tmp_data = [0 if x.strip() == "{}" else 1 for x in f.readlines()]
    all_data.extend(tmp_data)
data['rep']=all_data
print(np.mean(all_data))
len(all_data)

all_data = []
# for fpath in sorted(glob('sentiment/sentiment/*.sbertscore')):
for fpath in sorted(glob(f'{pos_file}/*.sbertscore')+glob(f'{neg_file}/*.sbertscore')):
    with open(fpath , 'r') as f:
        raw_data = f.readlines()
        tmp_data = [float(x.strip()) for x in raw_data[1:]]
    all_data.extend(tmp_data)
data['bert']=all_data
print(np.mean(all_data))

## dist-3
dist3_metrics=[]
for fpath in sorted(glob(f'{pos_file}/outputs_epsilon*.txt')+glob(f'{neg_file}/outputs_epsilon*.txt')):

    outputs=pd.read_json(fpath, lines=True)
    _,_,dist3=distinctness(outputs)
    dist3_metrics.append(dist3)
print(np.mean(dist3_metrics))

all_data = []
for fpath in sorted(glob(f'{pos_file}/*.formality_ext')):
    
    with open(fpath , 'r') as f:
        raw_data = f.readlines()
        tmp_data = [1 if float(x) >= 0.5 else 0 for x in raw_data]
    all_data.extend(tmp_data)
pos_constraint_acc = np.mean(tmp_data)
print(pos_constraint_acc)
    
for fpath in sorted(glob(f'{neg_file}/*.formality_ext')):
    
    with open(fpath , 'r') as f:
        raw_data = f.readlines()
        tmp_data = [1 if float(x) < 0.5 else 0 for x in raw_data]
    all_data.extend(tmp_data)
neg_constraint_acc = np.mean(tmp_data)
print(neg_constraint_acc)
    
data['formality_acc'] = all_data
print(np.mean(tmp_data))
print(np.mean(all_data))

tot_summ = data[['bert', 'formality_acc', 'ppl', 'fluency','rep']].mean().to_frame().T
tot_summ.insert(1, 'count', None)
tot_summ.insert(2, 'prop', None)
tot_summ.insert(4, 'formal', pos_constraint_acc)
tot_summ.insert(5, 'informal', neg_constraint_acc)
tot_summ.insert(8, 'dist-3', np.mean(dist3_metrics))
print(tot_summ)

data['sbert_geq_50']=data['bert'] >= 0.5
summ = data.groupby('sbert_geq_50')[['bert', 'formality_acc', 'ppl', 'fluency','rep',
       ]].mean().sort_index(ascending=False)
summ.insert(1, 'count', data.groupby('sbert_geq_50').size().sort_index(ascending=False))
summ.insert(2, 'prop', data.groupby('sbert_geq_50').size().sort_index(ascending=False)/data.shape[0])
summ.insert(4, 'formal', None)
summ.insert(5, 'informal', None)
summ.insert(8, 'dist-3', None)

print(summ)