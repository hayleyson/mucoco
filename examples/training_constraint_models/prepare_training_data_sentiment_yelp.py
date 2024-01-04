# -*- coding: utf-8 -*-
import os
import sys
import math
import logging
import json
sys.path.append("/home/s3/hyeryung/mucoco")
os.chdir("/home/s3/hyeryung/mucoco")
            
import pandas as pd

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
    
    data = data.sample(frac=1,random_state=999).reset_index(drop=True)#shuffle
    train_size = math.ceil(len(data) * 0.9)
    valid_size = math.ceil(len(data) * 0.05)

    train_data = data.iloc[:train_size,:].copy()
    valid_data = data.iloc[train_size:-valid_size, :].copy()
    test_data = data.iloc[-valid_size:,:].copy()

    ## save train/valid data for reproducibility
    os.makedirs('../data/yelp',exist_ok=True)
    train_data.to_json('../data/yelp/train.jsonl', lines=True, orient='records')
    valid_data.to_json('../data/yelp/valid.jsonl', lines=True, orient='records')
    test_data.to_json('../data/yelp/test.jsonl', lines=True, orient='records')

    

if __name__ == "__main__":

    
    main()