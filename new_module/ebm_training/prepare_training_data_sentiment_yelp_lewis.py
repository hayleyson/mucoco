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

    ### data load
    with open('data/sentiment/language-style-transfer/data/yelp/sentiment.train.0', 'r') as f:
        train_data_0 = [line.rstrip('\n') for line in f.readlines()]
    with open('data/sentiment/language-style-transfer/data/yelp/sentiment.train.1', 'r') as f:
        train_data_1 = [line.rstrip('\n') for line in f.readlines()]
    with open('data/sentiment/language-style-transfer/data/yelp/sentiment.dev.0', 'r') as f:
        dev_data_0 = [line.rstrip('\n') for line in f.readlines()]
    with open('data/sentiment/language-style-transfer/data/yelp/sentiment.dev.1', 'r') as f:
        dev_data_1 = [line.rstrip('\n') for line in f.readlines()]
    
    train_data = pd.DataFrame({"text": train_data_0 + train_data_1, "labels": [0] * len(train_data_0) + [1] * len(train_data_1)})
    valid_data = pd.DataFrame({"text": dev_data_0 + dev_data_1, "labels": [0] * len(dev_data_0) + [1] * len(dev_data_1)})

    ## sample ~ 5000 rows for valid_data
    valid_data = valid_data.groupby('labels').sample(2500, random_state=999)
    # _, valid_data = train_test_split(valid_data, test_size=5000, stratify=valid_data['labels'], random_state=999)
    print(f"train_data.shape: {train_data.shape}")
    print(f"train_data.labels.value_counts(): {train_data.labels.value_counts()}")
    print(f"valid_data.shape: {valid_data.shape}")
    print(f"valid_data.labels.value_counts(): {valid_data.labels.value_counts()}")

    ## save train/valid data for reproducibility
    data_dir='data/sentiment/language-style-transfer/data/yelp_for_training'
    os.makedirs(data_dir,exist_ok=True)
    train_data.to_json(f'{data_dir}/train.jsonl', lines=True, orient='records')
    valid_data.to_json(f'{data_dir}/valid.jsonl', lines=True, orient='records')
    

if __name__ == "__main__":

    
    main()