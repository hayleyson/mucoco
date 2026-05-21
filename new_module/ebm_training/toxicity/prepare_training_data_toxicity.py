# -*- coding: utf-8 -*-
import os
import sys
import math
import logging
import json





            
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                   datefmt="%m/%d/%Y %H:%M:%S", 
                   level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    pass

def binarize():
    
    data_dir='data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained'
    train_data = pd.read_json(f'{data_dir}/train.jsonl', lines=True)
    valid_data = pd.read_json(f'{data_dir}/dev.jsonl', lines=True)
    test_data = pd.read_json(f'{data_dir}/test.jsonl', lines=True)
    
    ## rename
    train_data = train_data.rename(columns={'toxicity': 'labels'})
    valid_data = valid_data.rename(columns={'toxicity': 'labels'})
    test_data = test_data.rename(columns={'toxicity': 'labels'})
    
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
    train_data['labels'] = train_data['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    valid_data['labels'] = valid_data['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    test_data['labels'] = test_data['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    
    ## count per label
    print(f"train_data: {len(train_data)}, 1: {len(train_data[train_data['labels'] == 1])}, 0: {len(train_data[train_data['labels'] == 0])}")
    print(f"valid_data: {len(valid_data)}, 1: {len(valid_data[valid_data['labels'] == 1])}, 0: {len(valid_data[valid_data['labels'] == 0])}")
    print(f"test_data: {len(test_data)}, 1: {len(test_data[test_data['labels'] == 1])}, 0: {len(test_data[test_data['labels'] == 0])}")
    
    train_data.to_json(f'{data_dir}/train_binary.jsonl', lines=True, orient='records')
    valid_data.to_json(f'{data_dir}/valid_binary.jsonl', lines=True, orient='records')
    test_data.to_json(f'{data_dir}/test_binary.jsonl', lines=True, orient='records')
    
def cleanup_data():
    
    data_dir='data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained'
    train_data = pd.read_json(f'{data_dir}/train.jsonl', lines=True)
    valid_data = pd.read_json(f'{data_dir}/old/dev.jsonl', lines=True)
    test_data = pd.read_json(f'{data_dir}/test.jsonl', lines=True)
    
    ## rename
    train_data = train_data.rename(columns={'toxicity': 'labels'})
    valid_data = valid_data.rename(columns={'toxicity': 'labels'})
    test_data = test_data.rename(columns={'toxicity': 'labels'})
    
    ## remove rows in test_data that has null labels
    print("shape before dropping nulls: ", test_data.shape)
    test_data = test_data.loc[~test_data['labels'].isna()].copy()
    print("shape after dropping nulls: ", test_data.shape)
    
    train_data.to_json(f'{data_dir}/train.jsonl', lines=True, orient='records')
    valid_data.to_json(f'{data_dir}/valid.jsonl', lines=True, orient='records')
    test_data.to_json(f'{data_dir}/test.jsonl', lines=True, orient='records')

if __name__ == "__main__":

    # main()
    # binarize()
    cleanup_data()
    