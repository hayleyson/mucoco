# -*- coding: utf-8 -*-
import os
import sys
import math
import logging
import json
# sys.path.append("/home/s3/hyeryung/mucoco")
# os.chdir("/home/s3/hyeryung/mucoco")
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.." )
sys.path.append(project_dir)
os.chdir(project_dir)
print("project_dir: ", project_dir)
            
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                   datefmt="%m/%d/%Y %H:%M:%S", 
                   level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def scale_labels_pt16(x):
    return (x + 3.) / 6. # range: -3 ~ 3 -> 0 ~ 1

def main():
    ## data load
    data= pd.read_csv('data/formality/PT16/answers', delimiter='\t', names=['score', 'individual scores', 'na', 'text'])
    data = data.sample(frac=1,random_state=999).reset_index(drop=True)#shuffle
    train_size = math.ceil(len(data) * 0.9)

    train_data = data.iloc[:train_size,:].copy()
    valid_data = data.iloc[train_size:, :].copy()

    if filtering: #only filter training data
        train_data['std'] = train_data['individual scores'].apply(lambda x: std([float(i) for i in str(x).split(',')]))
        train_data = train_data.loc[train_data['std'] < 1.5].copy()
        del train_data['std']

    del train_data['individual scores']
    del train_data['na']
    del valid_data['individual scores']
    del valid_data['na']
    
    train_data = train_data.rename(columns={"score": "labels"})
    train_data['labels'] = train_data['labels'].apply(lambda x: scale_labels_pt16)
    valid_data = valid_data.rename(columns={"score": "labels"})
    valid_data['labels'] = valid_data['labels'].apply(lambda x: scale_labels_pt16)

    ## save train/valid data for reproducibility
    if filtering:
        train_data.to_csv('data/formality/PT16/train_filtered.tsv', sep='\t', index=False)
    else:
        train_data.to_csv('data/formality/PT16/train.tsv', sep='\t', index=False)
    valid_data.to_csv('data/formality/PT16/valid.tsv', sep='\t', index=False)

def binarize():
    
    data_dir='data/formality/PT16'
    train_data = pd.read_csv(f'{data_dir}/train.tsv', sep='\t')
    valid_data = pd.read_csv(f'{data_dir}/valid.tsv', sep='\t')
    train_data = train_data.rename(columns={"score": "labels"})
    valid_data = valid_data.rename(columns={"score": "labels"})
    
    ## count
    print(f"train_data: {len(train_data)}")
    print(train_data.labels.value_counts())
    print(f"valid_data: {len(valid_data)}")
    print(valid_data.labels.value_counts())
    
    ## drop neutral
    train_data = train_data[train_data['labels'] != 0.5].copy()
    valid_data = valid_data[valid_data['labels'] != 0.5].copy()
    
    ## binarize
    train_data['labels'] = train_data['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    valid_data['labels'] = valid_data['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    
    ## count per label
    print(f"train_data: {len(train_data)}, 1: {len(train_data[train_data['labels'] == 1])}, 0: {len(train_data[train_data['labels'] == 0])}")
    print(f"valid_data: {len(valid_data)}, 1: {len(valid_data[valid_data['labels'] == 1])}, 0: {len(valid_data[valid_data['labels'] == 0])}")
    
    train_data.to_csv(f'{data_dir}/train_binary.tsv', sep='\t', index=False)
    valid_data.to_csv(f'{data_dir}/valid_binary.tsv', sep='\t', index=False)

if __name__ == "__main__":

    # main()
    binarize()
    