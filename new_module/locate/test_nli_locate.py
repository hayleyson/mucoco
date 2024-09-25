"""
SNLI 데이터에 대해서 locate을 실행해본 코드 (데이터셋 전체에 대해서 실행)
locate한 부분을 mask해서 그러고 났을 때 contradiction probability가 어떻게 달라지는지를 
locate accuracy 대신 써보려고 시도했었다. 
그런데 그 방식이 엄밀하게 locate accuracy를 대신할 수 있는가에 대해서는 고민이 더 필요한 상황임.

"""
import re
import random
import json
from typing import List

import wandb
import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix as cm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from new_module.locate.new_locate_utils import LocateMachine
from new_module.em_training.nli.models import EncoderModel
from new_module.em_training.nli.data_handling import load_nli_data, load_nli_test_data, NLI_Dataset, NLI_DataLoader

def merge_masks(input_text):
    
    ## re를 이용해 [MASK]의 반복을 찾고 이를 하나로 대체한다.
    # mask_start_end_indices = re.findall("(<mask>)+", input_text) 
    # for mask_index in mask_start_end_indices:
    #     input_text[: mask_index[0]] + "<mask>"
    # return re.sub("(<mask>)+", "<mask>", input_text)
    return re.sub("(<mask>)+", "...", input_text)

# locate 이 잘 되었는지 판단하기 위해서,
# 0) 라벨이 contradict인 샘플에 대해서 
# 1) locate된 곳을 [MASK]로 대체함 -- 연속된 [MASK]는 1개로 합침
#      [MASK] 대신에 삭제 또는 ..., random word로 대체도 가능함 
# 2) 별도의 NLI 모델을 써서 분류를 했을 때 contradict일 확률이 얼마나 내려가는지 확인 (특히 neutral일 확률이 높아져야 할것 같은데 과연 그럴지?)

# ----------------------------------- # 
# ARGS
batch_size = 64
max_num_tokens = 100

# ----------------------------------- # 
# nli dataset load
# nli model load

device = "cuda" if torch.cuda.is_available() else "cpu"

## nli 모델을 다른 것을 시도도 해보기 
nli_model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
nli_model = nli_model.to(device)
label_id = 2

# ----------------------------------- # 
## 학습에 사용한 eval dataset을 이용해서 평가하는 script를 추가한다.
## 그리고, 어제 생각한 것처럼 SNLI, MNLI, ANLI 별로 다르게 점수를 내 보는 것도 좋을 것 같다. 

## 이렇게 했을 때 문제점이 뭘까? SNLI에서는 못하고 ANLI는 잘하거나, SNLI는 잘하고 ANLI는 못하는 모델이 있을 수 있다.
## 그럴 경우, SNLI로만 비교하고 줄 세우는 것이 효과가 없을 수도 있다.
## 또한, preliminary하게 봤을 때 SNLI가 변별력이 없는 문제가 있었다.
## 다만 걱정되는 점은 ANLI는 데이터 자체가 어렵기 때문에 이 데이터에 대한 모델의 분류를 믿기 어려울 수도 있을 것 같다.
## 대체제로 생각해봄직 한것은, LLM을 classifier로 쓰는 것이다.

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

# nli_dataset = load_dataset('stanfordnlp/snli')
train_dev_data = load_nli_data(output_file_path="/data/hyeryung/mucoco/data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl")
dev_data = train_dev_data.loc[train_dev_data['split'] == 'dev']
nli_dataset = dev_data.to_dict(orient="records")

contradiction_indexes = [i for i, x in enumerate(nli_dataset) if x['original_labels'] == 2]
# neutral_indexes = [i for i, x in enumerate(nli_dataset['test']['label']) if x == 1]
print(f"# of contradiction data: {len(contradiction_indexes)}")
# print(f"# of neutral data: {len(neutral_indexes)}")

contradiction_data = NLIDataset(nli_dataset, contradiction_indexes)
# neutral_data = NLIDataset(nli_dataset['test'], neutral_indexes)

print(f"# of contradiction data: {len(contradiction_data)}")
# print(f"# of neutral data: {len(neutral_data)}")

contra_dataloader = DataLoader(contradiction_data, batch_size=batch_size, collate_fn = collate_fn, shuffle=False)
# neutral_dataloader = DataLoader(neutral_data, batch_size=batch_size, collate_fn = collate_fn)

### energy model load

def main(run_id, save_df=False):
    api = wandb.Api()
    run = api.run(run_id)
    config = run.config

    model_path = config['model_path']
    print(f"model_path: {model_path}")

    # config = load_config('new_module/em_training/config.yaml')
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncoderModel(config)
    model = model.to(config['device'])
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(e)
        print("try loading _pearsonr.pth")
        try:
            model_path = config['model_path'].split('.pth')[0] + '_pearsonr.pth'
            config['model_path'] = model_path
            model.load_state_dict(torch.load(model_path))
        except:
            return pd.DataFrame({"run_id": [run_id], 
                               "loss": [config['energynet']['loss']],
                               "label_column": [config['energynet']['label_column']],
                               "add_train_data": [config['energynet'].get('add_train_data', 'false')],
                               "add_ranking_loss": [config['energynet'].get('add_ranking_loss', 'false')],
                               "add_ranking_loss_setting": [config['energynet'].get('add_ranking_loss_setting', 'n/a')],
                               "batch_size": [config['energynet'].get('batch_size', 'n/a')],
                               "max_lr": [config['energynet'].get('max_lr', 'n/a')],
                               "optimizer": [config['energynet'].get('optimizer', 'adamw')],
                               "average contradiction proba before masking": ['nan'], 
                               "average contradiction proba after masking": ['nan'], 
                               "entail proportion after masking": ['nan'],
                               "neutral proportion after masking": ['nan'],
                               "contradiction proportion after masking": ['nan']})


    class Args:
        task = 'nli'

    locator = LocateMachine(model, model.tokenizer, Args())

    
    # ----------------------------------- #
    # Inference
    # original_class = "contradiction"
    contra_original = []
    contra_masked = []
    energys_before_masking = []
    energys_after_masking = []
    classes_before_masking = []
    classes_after_masking = []
    # loader = contra_dataloader if original_class == "contradiction" else neutral_dataloader
    loader = contra_dataloader
    for batch in tqdm.tqdm(loader):
        
        result = locator.locate_main(batch, 'grad_norm', max_num_tokens = max_num_tokens, unit="word", label_id=config['energynet']['energy_col'])

        print(result)

        ## post processing : merge multiple [MASK] into one 
        result = [merge_masks(x) for x in result]
        
        print('-' * 50)
        print(result)
        ##
        contra_masked.extend(result)
        
        tokens = nli_tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        tokens = tokens.to(device)
        texts = nli_tokenizer.batch_decode(tokens['input_ids'], skip_special_tokens=False)
        contra_original.extend(texts)
        with torch.no_grad():
            outputs = nli_model(**tokens)

        probas_before_masking = torch.softmax(outputs.logits, dim=1)
        probas_before_masking_energy = probas_before_masking[:, label_id]
        class_before_masking = torch.argmax(probas_before_masking,dim=-1)
        
        tokens_after_masking = nli_tokenizer(result, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)
        tokens_after_masking = tokens_after_masking.to(device)
        with torch.no_grad():
            outputs = nli_model(**tokens_after_masking)
        probas_after_masking =  torch.softmax(outputs.logits,dim=-1)
        probas_after_masking_energy = probas_after_masking[:, label_id]
        class_after_masking = torch.argmax(probas_after_masking,dim=-1)
        
        energys_before_masking.extend(probas_before_masking_energy.cpu().numpy())
        energys_after_masking.extend(probas_after_masking_energy.cpu().numpy())
        classes_before_masking.extend(class_before_masking.cpu().numpy())
        classes_after_masking.extend(class_after_masking.cpu().numpy())
        
        torch.cuda.empty_cache()

    # conf_matrix = cm(classes_before_masking, classes_after_masking, labels=[0,1,2])
    # pd.DataFrame(conf_matrix, columns=['Entailment', 'Neutral', 'Contradiction'], index=['Entailment', 'Neutral', 'Contradiction']).to_excel(f'snli_test_{original_class}_locating_cm_after_mask.xlsx')

    if save_df:
        pd.DataFrame({"original": contra_original, 
                "masked": contra_masked,
                "original_class": classes_before_masking,
                "masked_class": classes_after_masking,
                "original_contradiction_proba": energys_before_masking,
                "masked_contradiction_proba": energys_after_masking}).to_excel(f"snli_test_locating_result_{run_id.split('/')[-1]}.xlsx")

    # energy_dict = {'Average Contradiction Proba Before Masking': np.mean(energys_before_masking), 
    #                'Average Contradiction Proba After Masking': np.mean(energys_after_masking)}

    summary_df = pd.DataFrame({"run_id": [run_id], 
                               "loss": [config['energynet']['loss']],
                               "label_column": [config['energynet']['label_column']],
                               "add_train_data": [config['energynet'].get('add_train_data', 'false')],
                               "add_ranking_loss": [config['energynet'].get('add_ranking_loss', 'false')],
                               "add_ranking_loss_setting": [config['energynet'].get('add_ranking_loss_setting', 'n/a')],
                               "batch_size": [config['energynet'].get('batch_size', 'n/a')],
                               "max_lr": [config['energynet'].get('max_lr', 'n/a')],
                               "optimizer": [config['energynet'].get('optimizer', 'adamw')],
                               "average contradiction proba before masking": [np.mean(energys_before_masking)], 
                               "average contradiction proba after masking": [np.mean(energys_after_masking)], 
                               "entail proportion after masking": [(np.array(classes_after_masking)==0).sum()/len(classes_after_masking)],
                               "neutral proportion after masking": [(np.array(classes_after_masking)==1).sum()/len(classes_after_masking)],
                               "contradiction proportion after masking": [(np.array(classes_after_masking)==2).sum()/len(classes_after_masking)]})

    if save_df:
        summary_df.to_csv(f"snli_test_locating_result_{run_id.split('/')[-1]}_metrics.csv", index=False)
    # with open(f"snli_test_locating_result_{run_id.split('/')[-1]}_metrics.csv", 'w') as f:
        # 
        # f.write(f"average contradiction proba before masking, average contradiction proba after masking, entail proportion after masking, neutral proporation after masking, contradiction proportion after masking\n")
        # f.write(f"{np.mean(energys_before_masking)}, {np.mean(energys_after_masking)}, {(np.array(classes_after_masking)==0).sum()/len(classes_after_masking)}, {(np.array(classes_after_masking)==1).sum()/len(classes_after_masking)}, {(np.array(classes_after_masking)==2).sum()/len(classes_after_masking)} \n")
    return summary_df
        
if __name__ == "__main__":
    
#     run_ids = """4sr2xyod
# 3xorp8xg
# f33b5uy9
# 5xovvzpm
# twj4g8f9
# 8ct1dplo
# xiacbfxo
# saajhpve
# 25n5gs0a
# 3qhbpfvu
# i6u3mca5
# cq6zkeo8
# rejdn7cr
# t6lbryef
# xmoccelj
# 9hc3ufkv
# ksavttkv
# j2m2xtfu
# qjzbgnvp
# 9chwe2p3
# xohsica6
# cdpmq4y1
# 24i537i4""".split()
#     metrics_all = []
    
#     for run_id in run_ids:
#         metrics_all.append(main(f"hayleyson/nli_energynet/{run_id}"))
        
#     pd.concat(metrics_all).to_csv(f"snli_test_locating_result_metrics_merge_masks.csv", index=False)
    pass
        
        
    