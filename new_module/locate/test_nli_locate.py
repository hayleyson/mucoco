"""
SNLI 데이터에 대해서 locate을 실행해본 코드 (데이터셋 전체에 대해서 실행)
locate한 부분을 mask해서 그러고 났을 때 contradiction probability가 어떻게 달라지는지를 
locate accuracy 대신 써보려고 시도했었다. 
그런데 그 방식이 엄밀하게 locate accuracy를 대신할 수 있는가에 대해서는 고민이 더 필요한 상황임.

"""

import random
import json
from typing import List

import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix as cm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from new_module.locate.new_locate_utils import LocateMachine

# ----------------------------------- # 
# ARGS
batch_size = 64
max_num_tokens = 100
label_id = 2

# ----------------------------------- # 
# nli dataset load
# nli model load

device = "cuda" if torch.cuda.is_available() else "cpu"

nli_model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_model = nli_model.to(device)

nli_dataset = load_dataset('stanfordnlp/snli')
print(nli_dataset)


class Args:
    task = 'nli'

locator = LocateMachine(nli_model, nli_tokenizer, Args())

# ----------------------------------- # 
# Define dataset and dataloader
class NLIDataset(Dataset):
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = indexes
        
    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]
    
    def __getitems__(self, idxes:List[int]):
        return [self.dataset[self.indexes[j]] for j in idxes]
    
    def __len__(self):
        return len(self.indexes)
    
def collate_fn(examples):
    return [(x['premise'], x['hypothesis']) for x in examples]

contradiction_indexes = [i for i, x in enumerate(nli_dataset['test']['label']) if x == 2]
neutral_indexes = [i for i, x in enumerate(nli_dataset['test']['label']) if x == 1]
print(f"# of contradiction data: {len(contradiction_indexes)}")
print(f"# of neutral data: {len(neutral_indexes)}")

contradiction_data = NLIDataset(nli_dataset['test'], contradiction_indexes)
neutral_data = NLIDataset(nli_dataset['test'], neutral_indexes)

print(f"# of contradiction data: {len(contradiction_data)}")
print(f"# of neutral data: {len(neutral_data)}")

contra_dataloader = DataLoader(contradiction_data, batch_size=batch_size, collate_fn = collate_fn)
neutral_dataloader = DataLoader(neutral_data, batch_size=batch_size, collate_fn = collate_fn)

# ----------------------------------- #
# Inference
original_class = "neutral"
contra_original = []
contra_masked = []
energys_before_masking = []
energys_after_masking = []
classes_before_masking = []
classes_after_masking = []
loader = contra_dataloader if original_class == "contradiction" else neutral_dataloader
for batch in tqdm.tqdm(loader):
    
    result = locator.locate_main(batch, 'grad_norm', max_num_tokens = max_num_tokens, unit="word", label_id=label_id)

    contra_masked.extend(result)
    
    tokens = nli_tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
    tokens = tokens.to(device)
    texts = nli_tokenizer.batch_decode(tokens['input_ids'], skip_special_tokens=False)
    contra_original.extend(texts)
    with torch.no_grad():
        outputs = nli_model(**tokens)

    probas_before_masking = torch.softmax(outputs.logits, dim=1)
    probas_before_masking_energy = - probas_before_masking[:, label_id]
    class_before_masking = torch.argmax(probas_before_masking,dim=-1)
    
    tokens_after_masking = nli_tokenizer(result, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)
    tokens_after_masking = tokens_after_masking.to(device)
    with torch.no_grad():
        outputs = nli_model(**tokens_after_masking)
    probas_after_masking =  torch.softmax(outputs.logits,dim=-1)
    probas_after_masking_energy = - probas_after_masking[:, label_id]
    class_after_masking = torch.argmax(probas_after_masking,dim=-1)
    
    energys_before_masking.extend(probas_before_masking_energy.cpu().numpy())
    energys_after_masking.extend(probas_after_masking_energy.cpu().numpy())
    classes_before_masking.extend(class_before_masking.cpu().numpy())
    classes_after_masking.extend(class_after_masking.cpu().numpy())
    
    torch.cuda.empty_cache()

conf_matrix = cm(classes_before_masking, classes_after_masking, labels=[0,1,2])
pd.DataFrame(conf_matrix, columns=['Entailment', 'Neutral', 'Contradiction'], index=['Entailment', 'Neutral', 'Contradiction']).to_excel(f'snli_test_{original_class}_locating_cm_after_mask.xlsx')

pd.DataFrame({"original": contra_original, 
              "masked": contra_masked,
              "original_class": classes_before_masking,
              "masked_class": classes_after_masking,
              "original_energy": energys_before_masking,
              "masked_energy": energys_after_masking}).to_excel(f'snli_test_{original_class}_locating_text.xlsx')

energy_dict = {'Average Energy Before Masking': np.mean(energys_before_masking), 
               'Average Energy After Masking': np.mean(energys_after_masking)}
with open(f'snli_test_{original_class}_locating_energy.json', 'w') as f:
    
    json.dump(energy_dict, f)
    f.write('\n')