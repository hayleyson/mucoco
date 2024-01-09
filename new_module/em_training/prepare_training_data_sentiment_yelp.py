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

def main():

    ### data load
    data = pd.read_json('../data/yelp_academic_dataset_review.json', lines=True)

    del data['review_id']
    del data['user_id']
    del data['business_id']
    del data['useful']
    del data['funny']
    del data['cool']
    del data['date']
    
    data = data.rename(columns={'stars': 'labels'})
    
    print(f'labels min: {data.labels.min()}, labels max: {data.labels.max()}')
    data['labels'] = data['labels'].apply(lambda x: (x - 1.)/4. )
    print(f'labels min: {data.labels.min()}, labels max: {data.labels.max()}')
        
    ## stratified split, small valid set
    train_data, test_data = train_test_split(data, test_size=0.05, random_state=999,stratify=data['labels'])
    train_data, valid_data = train_test_split(train_data, test_size=5000, random_state=999,stratify=train_data['labels'])

    ## save train/valid data for reproducibility
    data_dir='../data/yelp'
    os.makedirs(data_dir,exist_ok=True)
    train_data.to_json(f'{data_dir}/train.jsonl', lines=True, orient='records')
    valid_data.to_json(f'{data_dir}/valid.jsonl', lines=True, orient='records')
    test_data.to_json(f'{data_dir}/test.jsonl', lines=True, orient='records')
    
def binarize():
    
    data_dir='data/yelp'
    train_data = pd.read_json(f'{data_dir}/train.jsonl', lines=True)
    valid_data = pd.read_json(f'{data_dir}/valid.jsonl', lines=True)
    test_data = pd.read_json(f'{data_dir}/test.jsonl', lines=True)
    
    ## count
    print(f"train_data: {len(train_data)}")
    print(train_data.labels.value_counts())
    print(f"valid_data: {len(valid_data)}")
    print(valid_data.labels.value_counts())
    print(f"test_data: {len(test_data)}")
    print(test_data.labels.value_counts())
    
    ## drop neutral
    train_data = train_data[train_data['labels'] != 0.5].copy()
    valid_data = valid_data[valid_data['labels'] != 0.5].copy()
    test_data = test_data[test_data['labels'] != 0.5].copy()
    
    ## binarize
    train_data['labels'] = train_data['labels'].apply(lambda x: 1 if x >= 0.5 else 0)
    valid_data['labels'] = valid_data['labels'].apply(lambda x: 1 if x >= 0.5 else 0)
    test_data['labels'] = test_data['labels'].apply(lambda x: 1 if x >= 0.5 else 0)
    
    ## count per label
    print(f"train_data: {len(train_data)}, 1: {len(train_data[train_data['labels'] == 1])}, 0: {len(train_data[train_data['labels'] == 0])}")
    print(f"valid_data: {len(valid_data)}, 1: {len(valid_data[valid_data['labels'] == 1])}, 0: {len(valid_data[valid_data['labels'] == 0])}")
    print(f"test_data: {len(test_data)}, 1: {len(test_data[test_data['labels'] == 1])}, 0: {len(test_data[test_data['labels'] == 0])}")
    
    train_data.to_json(f'{data_dir}/train_binary.jsonl', lines=True, orient='records')
    valid_data.to_json(f'{data_dir}/valid_binary.jsonl', lines=True, orient='records')
    test_data.to_json(f'{data_dir}/test_binary.jsonl', lines=True, orient='records')
    

if __name__ == "__main__":

    
    # main()
    binarize()