# -*- coding: utf-8 -*-
import os
import sys
import math
import logging
import json
sys.path.append("/home/s3/hyeryung/mucoco")
os.chdir("/home/s3/hyeryung/mucoco")
            
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
    
    ## stratified split, small valid set
    train_data, test_data = train_test_split(data, test_size=0.05, random_state=999,stratify=data['labels'])
    train_data, valid_data = train_test_split(train_data, test_size=5000, random_state=999,stratify=train_data['labels'])

    ## save train/valid data for reproducibility
    os.makedirs('../data/yelp',exist_ok=True)
    train_data.to_json('../data/yelp/train.jsonl', lines=True, orient='records')
    valid_data.to_json('../data/yelp/valid.jsonl', lines=True, orient='records')
    test_data.to_json('../data/yelp/test.jsonl', lines=True, orient='records')

    

if __name__ == "__main__":

    
    main()