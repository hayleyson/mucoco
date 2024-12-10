"""
- estimate locate accuracy using NLI locate labels 

"""
import argparse
import os
from typing import List
from copy import deepcopy

import wandb
import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from scipy import stats

from new_module.locate.new_locate_utils import LocateMachine
from new_module.em_training.nli.models import EncoderModel

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
    
def collate_fn_with_labels(examples):
    return [(x['premise'], x['hypothesis'], x['gold_label']) for x in examples]

def collate_fn(examples):
    return [(x['premise'], x['hypothesis']) for x in examples]

def get_word_pred_indexes_from_token_pred_indexes(row, tok2word_col: str, token_pred_indexes_col: str):
    words_indexes = []
    for id in row[token_pred_indexes_col]:
        words_indexes.append(row[tok2word_col][str(id)])
    return sorted(list(set(words_indexes)))

def get_word_pred_from_word_pred_indexes(row, words_col: str, word_pred_indexes_col: str):
    return [1 if i in row[word_pred_indexes_col] else 0 for i in range(len(row[words_col]))]

def get_pred_scores_word(row,words_col:str, token_pred_scores_col: str, word2tok_col:str, method='sum'):
    return_list=[]
    if method=='sum':
        func=np.sum
    elif method=='max':
        func=np.max
    elif method=='mean':
        func=np.mean
    for word_id in range(len(row[words_col])):
        return_list.append(func(np.array(row[token_pred_scores_col])[row[word2tok_col][str(word_id)]]))
    return return_list

def apply_ap(row, binary_labels_col:str, pred_scores_col:str):

    if sum(row[binary_labels_col])==0:
        return np.nan
    else:
        return average_precision_score(row[binary_labels_col],row[pred_scores_col])
    
def apply_precision(row, binary_labels_col:str, binary_preds_col:str):

    return precision_score(row[binary_labels_col],row[binary_preds_col], zero_division=np.nan)

def apply_recall(row, binary_labels_col:str, binary_preds_col:str):

    return recall_score(row[binary_labels_col],row[binary_preds_col], zero_division=np.nan)

def apply_f1(row, binary_labels_col:str, binary_preds_col:str):

    return f1_score(row[binary_labels_col],row[binary_preds_col], zero_division=np.nan)

def rr(out, labels, k = 6): #implement mean reciprocal rank
    idx_array = stats.rankdata(-out, axis=-1, method='min')
    # print(idx_array)
    labels = np.where(labels==1)[0].astype(int)
    # print(labels)
    rank = np.take_along_axis(idx_array, labels, axis=-1)
    # print(rank)
    rr=1/rank.min() if rank.min() <= k else 0.
    return rr

def get_rr(row, binary_labels_col:str, pred_scores_col:str):
    """suffix should start with _"""
    if sum(row[binary_labels_col])==0:
        return np.nan
    else:
        return rr(np.array(row[pred_scores_col]),np.array(row[binary_labels_col]))

def get_locate_metrics(run_id, criterion, model_dir, contra_data, contra_dataloader, setting, save_results=True):
    
    print('===== Start evaluating locate metrics for:', run_id, ' =====')
    contra_data = deepcopy(contra_data)
    
    model_path=os.path.join(model_dir, f'best_model_{criterion}.pth')
    print('Model loaded from model_path: ', model_path)
    ## load config
    api = wandb.Api()
    run = api.run(run_id)
    config = run.config
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## load model
    model = EncoderModel(config)
    model = model.to(config['device'])
    model.load_state_dict(torch.load(model_path,weights_only=True),strict=False)
    model.eval()
    
    config['model_path'] = model_path
    ## define tokenizer
    tokenizer = model.tokenizer
    
    # define locator
    locator = LocateMachine(model, model.tokenizer, 'nli')

    # locate!
    pred_all = []
    pred_indexes_all = []
    pred_scores_all = []
    masked_texts_all = []
    for batch in tqdm.tqdm(contra_dataloader):        
        
        premises = [x[0] for x in batch]
        hypotheses = [x[1] for x in batch]
        sequences = [tokenizer.bos_token + p + tokenizer.sep_token + h + tokenizer.eos_token for p, h in zip(premises, hypotheses)]
        tokenized_sequences = tokenizer(sequences, add_special_tokens=False,padding=True, truncation=True, return_tensors='pt')
        tokenized_sequences = tokenized_sequences.to(config['device'])
        hypotheses_start_idxes = [list(x).index(2)+1 for x in tokenized_sequences['input_ids']]
        hypotheses_end_idxes = [hypotheses_start_idxes[i] + list(x[hypotheses_start_idxes[i]:]).index(2) for i, x in enumerate(tokenized_sequences['input_ids'])]
        hypotheses_lengths = [hypotheses_end_idxes[i]-hypotheses_start_idxes[i] for i in range(len(batch))]
        
        masked_texts, scores, locate_ixes = locator.locate_main(
                                    tokenized_sequences, 
                                    'grad_norm', 
                                    max_num_tokens = max_num_tokens, 
                                    unit="word", 
                                    label_id=config['energynet']['energy_col'], 
                                    tokenized_input=True,
                                    return_scores_and_indices=True,
                                    use_energy=use_energy_for_gradient)
        
        hypotheses_scores = [x.tolist()[hypotheses_start_idxes[i]:hypotheses_end_idxes[i]] for i, x in enumerate(scores)]
        pred_indexes = [list(np.array(x)-hypotheses_start_idxes[i]) for i, x in enumerate(locate_ixes)]
        pred = [[1 if j in pred_indexes[i] else 0 for j in range(hypotheses_lengths[i])] for i in range(len(batch))]
        
        pred_all.extend(pred)
        pred_indexes_all.extend(pred_indexes)
        pred_scores_all.extend(hypotheses_scores)
        masked_texts_all.extend(masked_texts)
        
    # save token & workd-level predictions in the dataframe
    contra_data['hypothesis_token_pred_binary'] = pred_all
    contra_data['hypothesis_token_pred_indexes'] = pred_indexes_all
    contra_data['hypothesis_token_pred_scores'] = pred_scores_all
    contra_data['hypothesis_word_pred_indexes'] = contra_data.apply(lambda x: get_word_pred_indexes_from_token_pred_indexes(x, 'hypothesis_tok2word', 'hypothesis_token_pred_indexes'), axis=1)
    contra_data['hypothesis_word_pred_binary'] = contra_data.apply(lambda x: get_word_pred_from_word_pred_indexes(x, 'hypothesis_words', 'hypothesis_word_pred_indexes'), axis=1)
    contra_data['hypothesis_word_pred_scores'] = contra_data.apply(lambda x: get_pred_scores_word(x, 'hypothesis_words', 'hypothesis_token_pred_scores', 'hypothesis_word2tok', method='max'),axis=1)

    # ### Calculate Word-level Metrics
    contra_data['ap_word']=contra_data.apply(lambda x: apply_ap(x,"hypothesis_word_labels_binary", "hypothesis_word_pred_scores"),axis=1)
    contra_data['rr_word']=contra_data.apply(lambda x: get_rr(x,"hypothesis_word_labels_binary", "hypothesis_word_pred_scores"),axis=1)
    contra_data['precision_word']=contra_data.apply(lambda x: apply_precision(x,"hypothesis_word_labels_binary", "hypothesis_word_pred_binary"),axis=1)
    contra_data['recall_word']=contra_data.apply(lambda x: apply_recall(x,"hypothesis_word_labels_binary", "hypothesis_word_pred_binary"),axis=1)
    contra_data['f1_word']=contra_data.apply(lambda x: apply_f1(x,"hypothesis_word_labels_binary", "hypothesis_word_pred_binary"),axis=1)

    ## Summary metric
    mrr = contra_data['rr_word'].mean()
    map_score =  contra_data['ap_word'].mean()
    precision = contra_data['precision_word'].mean()
    recall = contra_data['recall_word'].mean()
    f1 = contra_data['f1_word'].mean()

    # ### Calculate Token-level Metrics
    contra_data['ap_tokens']=contra_data.apply(lambda x: apply_ap(x,"hypothesis_token_labels_binary", "hypothesis_token_pred_scores"),axis=1)
    contra_data['rr_tokens']=contra_data.apply(lambda x: get_rr(x,"hypothesis_token_labels_binary", "hypothesis_token_pred_scores"),axis=1)
    contra_data['precision_tokens']=contra_data.apply(lambda x: apply_precision(x,"hypothesis_token_labels_binary", "hypothesis_token_pred_binary"),axis=1)
    contra_data['recall_tokens']=contra_data.apply(lambda x: apply_recall(x,"hypothesis_token_labels_binary", "hypothesis_token_pred_binary"),axis=1)
    contra_data['f1_tokens']=contra_data.apply(lambda x: apply_f1(x,"hypothesis_token_labels_binary", "hypothesis_token_pred_binary"),axis=1)

    ## Summary metric
    mrr_tokens = contra_data['rr_tokens'].mean()
    map_score_tokens =  contra_data['ap_tokens'].mean()
    precision_tokens = contra_data['precision_tokens'].mean()
    recall_tokens = contra_data['recall_tokens'].mean()
    f1_tokens = contra_data['f1_tokens'].mean()

    print("Metrics evaluated at words level")
    print(f"mrr: {mrr:.4f}")
    print(f"map: {map_score:.4f}")
    print(f"mean precision: {precision:.4f}")
    print(f"mean recall: {recall:.4f}")
    print(f"mean f1: {f1:.4f}")

    print("Metrics evaluated at tokens level")
    print(f"mrr: {mrr_tokens:.4f}")
    print(f"map: {map_score_tokens:.4f}")
    print(f"mean precision: {precision_tokens:.4f}")
    print(f"mean recall: {recall_tokens:.4f}")
    print(f"mean f1: {f1_tokens:.4f}")

    if save_results:
        if criterion is not None:
            raw_result_save_path = os.path.join(os.path.dirname(model_path), f'{setting}_locate_result_{criterion}_{"energy" if use_energy_for_gradient else "proba"}.jsonl')
            contra_data[['pairID', 'hypothesis_word_pred_binary', 'hypothesis_word_pred_scores', 'hypothesis_token_pred_binary', 'hypothesis_token_pred_scores']].to_json(raw_result_save_path, lines=True, orient='records')
        else:
            raw_result_save_path = os.path.join(os.path.dirname(model_path), f'{setting}_locate_result_{"energy" if use_energy_for_gradient else "proba"}.jsonl')
            contra_data[['pairID', 'hypothesis_word_pred_binary', 'hypothesis_word_pred_scores', 'hypothesis_token_pred_binary', 'hypothesis_token_pred_scores']].to_json(raw_result_save_path, lines=True, orient='records')
    
        print('Sample-level results saved at:', raw_result_save_path)
        metrics_path = os.path.join(os.path.dirname(model_path), f'{setting}_locate_metrics.csv')

        if not os.path.exists(metrics_path):
            pd.DataFrame({'run_id': [run_id],
                        'criterion': [criterion],
                        'use_energy_for_gradient':[use_energy_for_gradient],
                        # 'classification_accuracy': [acc],
                        'mrr_words': [mrr],
                        'map_words': [map_score],
                        'mean precision_words': [precision],
                        'mean recall_words': [recall],
                        'mean f1_words': [f1],
                        'mrr_tokens': [mrr_tokens], 
                        'map_tokens': [map_score_tokens],
                        'mean precision_tokens': [precision_tokens],
                        'mean recall_tokens': [recall_tokens],
                        'mean f1_tokens': [f1_tokens]}).to_csv(metrics_path,index=False)
        else:
            with open(metrics_path, 'a') as f:
                # f.write(f"{run_id},{criterion},{use_energy_for_gradient},{acc},{mrr},{map_score},{precision},{recall},{mrr_tokens},{map_score_tokens},{precision_tokens},{recall_tokens}\n")
                f.write(f"{run_id},{criterion},{use_energy_for_gradient},{mrr},{map_score},{precision},{recall},{f1},{mrr_tokens},{map_score_tokens},{precision_tokens},{recall_tokens},{f1_tokens}\n")
        
        print('Summary metrics saved at:', metrics_path)


# ARGS
use_energy_for_gradient = True
max_num_tokens = 100
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

    
if __name__ == "__main__":
    
    runpath2modelpath = \
        {'hayleyson/nli_energynet/9s1fli5s': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_3class_finegrained_labels_cross_entropy_n_a/9s1fli5s/'}
        # {'hayleyson/nli_energynet/u6tu4o9t': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_margin_ranking/1731247397', 
        # 'hayleyson/nli_energynet/msv6wq04': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_margin_ranking/1731569657',
        # 'hayleyson/nli_energynet/wxer9zw3': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_margin_ranking/1731654805',
        # 'hayleyson/nli_energynet/svk3b64y': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_margin_ranking/svk3b64y',
        # 'hayleyson/nli_energynet/2qbql1br': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_margin_ranking/2qbql1br',
        # 'hayleyson/nli_energynet/wdw0y1qp': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731247443',
        # 'hayleyson/nli_energynet/c4ll3opi': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/1731654539',
        # 'hayleyson/nli_energynet/nznwxbaw': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/nznwxbaw', 
        # 'hayleyson/nli_energynet/07m5lce9': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/07m5lce9',
        # 'hayleyson/nli_energynet/lkavms6l': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a/lkavms6l',
        # 'hayleyson/nli_energynet/ni8cu2nw': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_margin_ranking/1731247881',
        # 'hayleyson/nli_energynet/eiqzuowj': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_margin_ranking/1731651193',
        # 'hayleyson/nli_energynet/w6hmipfb': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_margin_ranking/w6hmipfb', 
        # 'hayleyson/nli_energynet/id06pp5n': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_margin_ranking/id06pp5n',
        # 'hayleyson/nli_energynet/mev8cuhp': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_margin_ranking/mev8cuhp',
        # 'hayleyson/nli_energynet/e8cse9ni': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_n_a/1731247889',
        # 'hayleyson/nli_energynet/qhhowe3e': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_n_a/1731568014',
        # 'hayleyson/nli_energynet/xie6veic': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_n_a/1731651194',
        # 'hayleyson/nli_energynet/auxxqz22': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_n_a/auxxqz22',
        # 'hayleyson/nli_energynet/ub4nku33': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_n_a/ub4nku33',
        # 'hayleyson/nli_energynet/zgs9e2sr': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/',
        # 'hayleyson/nli_energynet/gyzuycek': '/data/hyeryung/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/gyzuycek'
        # }
        
        
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default='nli_contra_300', help="dataset to use for evaluation")
    args = parser.parse_args()
    setting = args.setting
    
    if setting == 'nli_contra_300':
        # Load NLI locate data & define dataloader
        contra_data = pd.read_json('new_module/data/NLI_locate/nli_contra_300_locate_labels.jsonl', lines=True)

    elif setting == 'epr_snli':
        
        # Load NLI locate data & define dataloader
        contra_data = pd.read_json('new_module/data/EPR/text_file/snli_annotation/snli_locate_labels.jsonl', lines=True)
        contra_data = contra_data[contra_data['gold_label'] == 'contradiction'].copy()
        
    nli_dataset = contra_data.to_dict(orient="records")
    contradiction_indexes = list(range(len(nli_dataset)))
    contradiction_dataset = NLIDataset(nli_dataset, contradiction_indexes)
    contra_dataloader = DataLoader(contradiction_dataset, batch_size=batch_size, collate_fn = collate_fn, shuffle=False)
    print(f"# data used to evaluate: {len(contradiction_dataset)}")
        
    # run_id = 'hayleyson/nli_energynet/ub4nku33'
    # criterion = 'loss'
    # model_dir = runpath2modelpath[run_id]
    # get_locate_metrics(run_id, criterion, model_dir, contra_data, contra_dataloader, save_results=True)
    
    for run_id, model_dir in runpath2modelpath.items():
        for criterion in ['loss', 'pearsonr']:
            get_locate_metrics(run_id, criterion, model_dir, contra_data, contra_dataloader, setting, save_results=True)