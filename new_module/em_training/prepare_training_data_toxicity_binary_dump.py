# -*- coding: utf-8 -*-
## previously named prepare_training_data_toxicity_binary_dump.py
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
print("current_dir: ", os.getcwd())
            
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                   datefmt="%m/%d/%Y %H:%M:%S", 
                   level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    data_dir='/shared/s3/lab07/hyeryung/loc_edit/data'
    all_data = pd.read_csv(f'{data_dir}/all_data.csv')
    
    print(all_data.head())
    print(all_data.shape)
    print(all_data.columns)
    
    ## select columns relevant
    select_columns = ["comment_text", "toxicity"]
    all_data = all_data[select_columns].copy()    
    
    ## rename columns
    all_data = all_data.rename(columns={'comment_text': 'text', 'toxicity': 'labels'})
    print(all_data['labels'].describe())
    
    ## clean up data 
    # - 1. Dropped rows with duplicate text and toxicity label (21,713 rows among ~2,000,000 rows in total.)
    # - 2. Dropped rows with duplicate text with different toxicity labels (4,347 unique texts. Dropped 10,234 rows)
    print(f"Before dropping duplicates: {all_data.shape[0]}")
    print("Dropped rows with duplicate text and toxicity label..  ", end=" ")
    all_data = all_data.drop_duplicates(subset=['text', 'labels'], keep='first').copy()
    print(f": {all_data.shape[0]}")
    print("Dropped rows with duplicate text with different toxicity labels..")
    all_data = all_data.drop_duplicates(subset=['text'], keep=False).copy()
    print(f": {all_data.shape[0]}")
    
    ## set apart 5000 for test and 5000 for valid
    ## create a new column indicating bins of toxicity
    all_data['labels_bin'] = all_data['labels'].apply(lambda x: math.floor(x*10)/10)
    
    train_data, test_data = train_test_split(all_data, test_size=5000, random_state=42, stratify=all_data['labels_bin'])
    train_data, valid_data = train_test_split(train_data, test_size=5000, random_state=42, stratify=train_data['labels_bin'])
    
    print(f"size of train_data: {len(train_data)}, \
count of > 0.5 : {len(train_data.loc[train_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(train_data.loc[train_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(train_data.loc[train_data['labels'] == 0.5,:])}")
    
    print(f"size of valid_data: {len(valid_data)}, \
count of > 0.5 : {len(valid_data.loc[valid_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(valid_data.loc[valid_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(valid_data.loc[valid_data['labels'] == 0.5,:])}")
    
    print(f"size of test_data: {len(test_data)}, \
count of > 0.5 : {len(test_data.loc[test_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(test_data.loc[test_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(test_data.loc[test_data['labels'] == 0.5,:])}")
    
    train_data.to_json(f'{data_dir}/train_dump.jsonl', lines=True, orient='records')
    valid_data.to_json(f'{data_dir}/valid_dump.jsonl', lines=True, orient='records')
    test_data.to_json(f'{data_dir}/test_dump.jsonl', lines=True, orient='records')
     
    
    

def binarize():
    
    print("Create a binarized version of data")
    data_dir='/shared/s3/lab07/hyeryung/loc_edit/data'
    
    train_data = pd.read_json(f'{data_dir}/train_dump.jsonl', lines=True)
    valid_data = pd.read_json(f'{data_dir}/valid_dump.jsonl', lines=True)
    test_data = pd.read_json(f'{data_dir}/test_dump.jsonl', lines=True)
    
    print(f"size of train_data: {len(train_data)}, \
count of > 0.5 : {len(train_data.loc[train_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(train_data.loc[train_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(train_data.loc[train_data['labels'] == 0.5,:])}")
    
    print(f"size of valid_data: {len(valid_data)}, \
count of > 0.5 : {len(valid_data.loc[valid_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(valid_data.loc[valid_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(valid_data.loc[valid_data['labels'] == 0.5,:])}")
    
    print(f"size of test_data: {len(test_data)}, \
count of > 0.5 : {len(test_data.loc[test_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(test_data.loc[test_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(test_data.loc[test_data['labels'] == 0.5,:])}")
    
    ## remove samples that with label that is 0.5
    train_data = train_data[train_data['labels'] != 0.5].copy()
    valid_data = valid_data[valid_data['labels'] != 0.5].copy()
    test_data = test_data[test_data['labels'] != 0.5].copy()
    
    print("After dropping rows with labels=0.5")
    print(f"size of train_data: {len(train_data)}, \
count of > 0.5 : {len(train_data.loc[train_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(train_data.loc[train_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(train_data.loc[train_data['labels'] == 0.5,:])}")
    
    print(f"size of valid_data: {len(valid_data)}, \
count of > 0.5 : {len(valid_data.loc[valid_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(valid_data.loc[valid_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(valid_data.loc[valid_data['labels'] == 0.5,:])}")
    
    print(f"size of test_data: {len(test_data)}, \
count of > 0.5 : {len(test_data.loc[test_data['labels'] > 0.5,:])}, \
count of < 0.5 : {len(test_data.loc[test_data['labels'] < 0.5,:])}, \
count of =0.5 : {len(test_data.loc[test_data['labels'] == 0.5,:])}")
    
    ## binarize the labels 
    
    def binarize(x):
        if x >= 0.5:
            return 1.
        else:
            return 0.
    
    train_data['labels'] = train_data['labels'].apply(binarize)
    valid_data['labels'] = valid_data['labels'].apply(binarize)
    test_data['labels'] = test_data['labels'].apply(binarize)
    
    ## after binarizing the labels
    print("After binarizing")
    print(f"size of train_data: {len(train_data)}, \
count of = 1. : {len(train_data.loc[train_data['labels'] == 1.0,:])}, \
count of = 0. : {len(train_data.loc[train_data['labels'] == 0.0,:])}, \
count of =0.5 : {len(train_data.loc[train_data['labels'] == 0.5,:])}")
    
    print(f"size of valid_data: {len(valid_data)}, \
count of = 1. : {len(valid_data.loc[valid_data['labels'] == 1.0,:])}, \
count of = 0. : {len(valid_data.loc[valid_data['labels'] == 0.0,:])}, \
count of =0.5 : {len(valid_data.loc[valid_data['labels'] == 0.5,:])}")
    
    print(f"size of test_data: {len(test_data)}, \
count of = 1. : {len(test_data.loc[test_data['labels'] == 1.0,:])}, \
count of = 0. : {len(test_data.loc[test_data['labels'] == 0.0,:])}, \
count of =0.5 : {len(test_data.loc[test_data['labels'] == 0.5,:])}")
    
    train_data.to_json(f'{data_dir}/train_bin_dump.jsonl', lines=True, orient='records')
    valid_data.to_json(f'{data_dir}/valid_bin_dump.jsonl', lines=True, orient='records')
    test_data.to_json(f'{data_dir}/test_bin_dump.jsonl', lines=True, orient='records')
    
    
if __name__ == "__main__":

    # main()
    binarize()