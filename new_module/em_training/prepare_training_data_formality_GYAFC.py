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
print("current_dir: ", os.getcwd())
            
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                   datefmt="%m/%d/%Y %H:%M:%S", 
                   level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    ## data load
    
    with open('data/formality/GYAFC_Corpus/Entertainment_Music/train/informal' , 'r') as f:
        informal = [x.rstrip() for x in f.readlines()]
    with open('data/formality/GYAFC_Corpus/Entertainment_Music/train/formal' , 'r') as f:
        formal = [x.rstrip() for x in f.readlines()]
    train_data = pd.DataFrame({'text': informal + formal, 'labels': [0] * len(informal) + [1] * len(formal)})
    
    with open('data/formality/GYAFC_Corpus/Entertainment_Music/tune/informal' , 'r') as f:
        informal = [x.rstrip() for x in f.readlines()]
    with open('data/formality/GYAFC_Corpus/Entertainment_Music/tune/formal' , 'r') as f:
        formal = [x.rstrip() for x in f.readlines()]
    valid_data = pd.DataFrame({'text': informal + formal, 'labels': [0] * len(informal) + [1] * len(formal)})
    
    with open('data/formality/GYAFC_Corpus/Entertainment_Music/test/informal' , 'r') as f:
        informal = [x.rstrip() for x in f.readlines()]
    with open('data/formality/GYAFC_Corpus/Entertainment_Music/test/formal' , 'r') as f:
        formal = [x.rstrip() for x in f.readlines()]
    test_data = pd.DataFrame({'text': informal + formal, 'labels': [0] * len(informal) + [1] * len(formal)})
    
    print(f"size of train_data: {len(train_data)}, num formal: {len(train_data[train_data['labels'] == 1])}, num informal: {len(train_data[train_data['labels'] == 0])}")
    print(f"size of valid_data: {len(valid_data)}, num formal: {len(valid_data[valid_data['labels'] == 1])}, num informal: {len(valid_data[valid_data['labels'] == 0])}")
    print(f"size of test_data: {len(test_data)}, num formal: {len(test_data[test_data['labels'] == 1])}, num informal: {len(test_data[test_data['labels'] == 0])}")

    ## save train/valid data for reproducibility
    train_data.to_json('data/formality/GYAFC_Corpus/Entertainment_Music/train.jsonl', lines=True, orient='records')
    valid_data.to_json('data/formality/GYAFC_Corpus/Entertainment_Music/valid.jsonl', lines=True, orient='records')
    test_data.to_json('data/formality/GYAFC_Corpus/Entertainment_Music/test.jsonl', lines=True, orient='records')

if __name__ == "__main__":

    main()
    