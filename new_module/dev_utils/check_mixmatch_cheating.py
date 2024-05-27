## check if by just converting letter to all lowercase, informality score can increase.

import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def formality_score_ext(texts, device):
    
    
    class CustomDataset():
        def __init__(self, data_list):
            self.data_list = data_list
            
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, index):
            return self.data_list[index]

    def collate_fn(example_batch):
       return tokenizer(example_batch, padding=True, truncation=True, return_tensors="pt").to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/roberta-base-formality")
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/roberta-base-formality")
    model.to(device)
    model.eval()
    
    softmax = nn.Softmax(dim=-1)
    
    # generations_df = generations_df.explode('generations')
    # generations = generations_df["generations"]
    # texts = [example['text'] for example in generations]
    dataset = CustomDataset(texts)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=False, collate_fn=collate_fn)
    
    formality_scores = []
    formal_counts = 0
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            # print(outputs.logits)
            probs = softmax(outputs.logits)
            formality_scores.extend(probs[:, -1].tolist())
            formal_counts += torch.sum(torch.where(probs[:,-1] >= 0.5,1,0)).item()
            
    
    return np.nanmean(formality_scores), formal_counts/len(texts)

with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal', 'r') as f:
    data = [x.rstrip() for x in f.readlines()]
    
# dataset = Dataset.from_pandas(pd.DataFrame({'text': data}))
# classifer = pipeline('text-classification', 
#                      model='cointegrated/roberta-base-formality', 
#                      tokenizer='cointegrated/roberta-base-formality',
#                      batch_size=32)
# formal_counts = 0
# formal_proba = []
# for out in classifer(KeyDataset(dataset, "text")):
#     formal_counts+= 1 if out['label'] == 'LABEL_1' else 0
#     formal_proba.append(out['score'] if out['label'] == 'LABEL_1' else 1 - out['score'])
# print(formal_counts / len(data))
# print(np.mean(formal_proba))

print(formality_score_ext(data, device='cuda:0'))
# (0.9713083980876139, 0.9815157116451017)

data_lower = [x.lower() for x in data]

print(formality_score_ext(data_lower, device='cuda:0'))
# (0.24722898595794465, 0.23197781885397412)