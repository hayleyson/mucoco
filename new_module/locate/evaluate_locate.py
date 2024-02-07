#!/usr/bin/env python
# coding: utf-8
"""
Prototyping code for evaluating the accuracy of locating tokens to edit against ground truth.
For metrics, MRR (mean reciprocal rank),Average Precision,and F1 score is used.
Other candidate metrics include AUC and Recall @ K.
The unit of calculating the metric is "token" at the moment.
But it will expand to "character" and "word".
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, average_precision_score

def count_pad_token(x):
    return np.sum(np.array(x)==tokenizer.pad_token_id)
def remove_pad_token(x):
    return list(np.array(x)[np.array(x)!=tokenizer.pad_token_id])
def remove_label_for_pad_token(row, colname):
    return list(np.array(row[colname])[np.array(row['tokens'])!=tokenizer.pad_token_id])

def index2binary(row, index_colname='pred', len_colname='tokens'):
    return [1 if i in row[index_colname] else 0 for i in range(len(row[len_colname]))]


def apply_f1(row, suffix=''):
    """suffix should start with _"""
    return f1_score(row[f'labels_binary{suffix}'],row[f'pred_binary{suffix}'], zero_division=np.nan)
def apply_f2(row, suffix=''):
    """suffix should start with _"""
    return fbeta_score(row[f'labels_binary{suffix}'],row[f'pred_binary{suffix}'], beta=2, zero_division=np.nan)
def apply_ap(row, suffix=''):
    """suffix should start with _"""
    if sum(row[f'labels_binary{suffix}'])==0:
        return np.nan
    else:
        return average_precision_score(row[f'labels_binary{suffix}'],row[f'pred_scores{suffix}'])
def apply_precision(row, suffix=''):
    """suffix should start with _"""
    return precision_score(row[f'labels_binary{suffix}'],row[f'pred_binary{suffix}'], zero_division=np.nan)
def apply_recall(row, suffix=''):
    """suffix should start with _"""
    return recall_score(row[f'labels_binary{suffix}'],row[f'pred_binary{suffix}'], zero_division=np.nan)


def rr(out, labels, k = 6): #implement mean reciprocal rank
    idx_array = stats.rankdata(-out, axis=-1, method='min')
    # print(idx_array)
    labels = np.where(labels==1)[0].astype(int)
    # print(labels)
    rank = np.take_along_axis(idx_array, labels, axis=-1)
    # print(rank)
    rr=1/rank.min() if rank.min() <= k else 0.
    return rr
def get_rr(row, suffix=''):
    """suffix should start with _"""
    if sum(row[f'labels_binary{suffix}'])==0:
        return np.nan
    else:
        return rr(np.array(row[f'pred_scores{suffix}']),np.array(row[f'labels_binary{suffix}']))
    

def get_tok2char(row: pd.Series, dataset_type:str="gpt2") -> dict:
    """
    A function to convert a list of tokens into a mapping between each token's index and its corresponding character offsets.
    @param row: A row from dataframe
    @return tok2char: A dictionary with token's location index as keys and tuples of corresponding character offsets as values.

    Example:
    row=pd.Series()
    row['text']='wearing games and holy ****ing shit do I hate horse wearing games .'
    row['tokens']=[86, 6648, 1830, 290, 11386, 25998, 278, 7510, 466, 314, 5465, 8223, 5762, 1830, 764]
    tok2char=get_tok2char(row, "tsd")
    tok2char
    {0: (0,),
    1: (1, 2, 3, 4, 5, 6),
    2: (7, 8, 9, 10, 11, 12),
    3: (13, 14, 15, 16),
    ...
    13: (59, 60, 61, 62, 63, 64),
    14: (65,66)}
    """
    # if dataset_type == "gpt2":        
    #     tok2char=dict()
    #     token_offsets=[0]
    #     for i in range(1,len(row['tokens'])+1):
    #         decoded=tokenizer.decode(row['tokens'][:i])
    #         token_offsets.append(len(decoded))
    #         tok2char[i-1]=tuple(range(token_offsets[i-1],token_offsets[i]))
    #     return tok2char
    
    # elif dataset_type == "tsd":
    tok2char=dict()
    token_offsets=[0]
    j = 0
    for i in range(1,len(row['tokens'])+1):
        while True:
            if tokenizer.decode(tokenizer.encode(row['text'][:j],add_special_tokens=False)) != tokenizer.decode(row['tokens'][:i]):
                if tokenizer.decode(row['tokens'][:i])[-1]=='�':#handle a case where a character is split into multiple tokens
                    break
                j+=1
            else:
                token_offsets.append(j)
                tok2char[i-1]=tuple(range(token_offsets[-2],token_offsets[-1]))
                tmp_id = i-2
                while (tmp_id >= 0 and tmp_id not in tok2char):
                    tok2char[tmp_id]=tuple(range(token_offsets[-2],token_offsets[-1]))
                    tmp_id-=1
                j+=1
                break
    return tok2char

def get_word2char(row: pd.Series, ws: str) -> dict:
    """
    A function to convert a list of words into a mapping between each word's index and its corresponding character offsets.
    @param row: A row from dataframe
    @return word2char: A dictionary with word's location index as keys and tuples of corresponding character offsets as values.

    Caveat:
    This code assumes that words are separated by only one type of whitespace, e.g. space.

    Example:
    row=pd.Series()
    row['words']=['wearing', 'games', 'and', 'holy', '****ing', 'shit', 'do', 'I', 'hate', 'horse', 'wearing', 'games.']
    word2char=get_word2char(row)
    word2char
    {0: (0, 1, 2, 3, 4, 5, 6),
    1: (7, 8, 9, 10, 11, 12),...
    9: (45, 46, 47, 48, 49, 50),
    10: (51, 52, 53, 54, 55, 56, 57, 58),
    11: (59, 60, 61, 62, 63, 64, 65)}
    """
    
    word_offsets=[0]
    word2char=dict()
    for i in range(1,len(row['words'])+1):
        decoded=ws.join(row['words'][:i])
        word_offsets.append(len(decoded))
        word2char[i-1]=tuple(range(word_offsets[i-1],word_offsets[i]))
    return word2char

## group token indices that belong to the same word

def get_word2tok(row: pd.Series, ws: str=None) -> dict:
    """
    A function that take a list of words and a corresponding list of tokens 
    into a mapping between each word's index and its corresponding token indexes.
    @param row: A row from dataframe
    @return word2char: A dictionary with word's location index as keys and tuples of corresponding token location indexes as values.

    Example:
    row=pd.Series()
    row['words']=['wearing', 'games', 'and', 'holy', '****ing', 'shit', 'do', 'I', 'hate', 'horse', 'wearing', 'games.']
    row['tokens']=[86, 6648, 1830, 290, 11386, 25998, 278, 7510, 466, 314, 5465, 8223, 5762, 1830, 13]
    word2tok=get_word2tok(row)
    word2tok
    {0: [0, 1],
    1: [2],
    2: [3],
    ...
    10: [12],
    11: [13, 14]}
    """
    global tokenizer
    
    jl, jr, k = 0, 0, 0
    grouped_tokens = []
    if ws is not None:
        while jr <= len(row['tokens'])+1 and k < len(row['words']):
            # print(f"{jl}, {jr}, {k}: {tokenizer.decode(row['tokens'][jl:jr]).strip(' ')}")
            if tokenizer.decode(row['tokens'][jl:jr]).strip(' ') == row['words'][k]:
                grouped_tokens.append(list(range(jl,jr)))
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1
        word2tok = dict(zip(range(len(grouped_tokens)), grouped_tokens))
    else:
        while jr <= len(row['tokens'])+1 and k < len(row['words']):
            # print(f"{jl}, {jr}, {k}: {tokenizer.decode(row['tokens'][jl:jr]).strip()}")
            if tokenizer.decode(row['tokens'][jl:jr]).strip() == row['words'][k]:
                grouped_tokens.append(list(range(jl,jr)))
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1
        word2tok = dict(zip(range(len(grouped_tokens)), grouped_tokens))
    return word2tok

def kv_swap(x):

    return_dict=dict()
    for k,v in x.items():
        for item in v:
            return_dict[item]=k
    return return_dict

def get_pred_word(row):

    return sorted(list(set([row['tok2word'][id] for id in row['pred']])))
def get_pred_binary_word(row):

    return [1 if id in row['pred_word'] else 0 for id in range(len(row['words']))]

def get_labels_binary_word(row, dataset_type="gpt2"):

    if dataset_type == "gpt2":
        labels_token_index = np.where(np.array(row['labels_binary'])==1)[0]
        labels_word_index = list(set([row['tok2word'][id] for id in labels_token_index]))
    elif dataset_type == "tsd":
        labels_char_index = np.array(row['labels_index_char'])
        labels_word_index = list(set([row['char2word'][id] for id in labels_char_index]))

    return [1 if id in labels_word_index else 0 for id in range(len(row['words']))]

def get_labels_binary_char(row, dataset_type="gpt2"):

    if dataset_type == "gpt2":
        labels_token_index = np.where(np.array(row['labels_binary'])==1)[0]
        labels_char_index = list(set(sum([list(row['tok2char'][id]) for id in labels_token_index],[])))
        return [1 if id in labels_char_index else 0 for id in range(len(row['char']))]
    elif dataset_type == "tsd": ## for tsd, this function is not needed.
        raise NotImplementedError
    
def get_labels_binary_token(row, dataset_type="gpt2"):

    if dataset_type == "gpt2": ## for gpt2, this function is not needed.
        raise NotImplementedError
    elif dataset_type == "tsd":
        labels_char_index = np.array(row['labels_index_char'])
        labels_token_index = list(set([row['char2tok'][id] for id in labels_char_index]))
        return [1 if id in labels_token_index else 0 for id in range(len(row['tokens']))]


def get_pred_char(row):
    
    return sorted(list(set(sum([list(row['tok2char'][id]) for id in row['pred']],[]))))

def get_pred_binary_char(row):

    return [1 if id in row['pred_char'] else 0 for id in range(len(row['char']))]



def get_pred_scores_char(row):
    """
    Each character gets the score of the token it belongs.
    """
    return_list=[]
    for char_id in range(len(row['char'])):
        try:
            return_list.append(row['pred_scores'][row['char2tok'][char_id]])
        except:
            print(row['text'])
            print(len(row['char']))
            print(len(row['char2tok'].keys()))
    return return_list

def get_pred_scores_word(row,method='sum'):
    return_list=[]
    if method=='sum':
        func=np.sum
    elif method=='max':
        func=np.max
    elif method=='mean':
        func=np.mean
    for word_id in range(len(row['words'])):
        return_list.append(func(np.array(row['pred_scores'])[row['word2tok'][word_id]]))
    return return_list


def calculate_locate_metrics(predictions_labels:pd.DataFrame, args: argparse.Namespace) -> None:
    
    data_stats_f = open("_".join([os.path.splitext(args.pred_file_path)[0],"data_stats.txt"]), "w")
    summary_result=dict()

    if 'tokens' not in predictions_labels.columns:
        predictions_labels['tokens']=predictions_labels['text'].apply(lambda x: tokenizer.encode(x,add_special_tokens=False))
    # ### Token ↔︎ Word ↔︎ Char 이 가능한 mapping 정의
    sample_text = predictions_labels[['text','tokens']].copy()
    sample_text['char']=sample_text['text'].apply(list)
    sample_text['char_index']=sample_text['char'].apply(lambda x: list(range(len(x))))
    assert (sample_text['char'].apply(len) != sample_text['char_index'].apply(len)).sum() == 0

    sample_text['tokens_index']=sample_text['tokens'].apply(lambda x: list(range(len(x))))

    # if args.dataset_type == "gpt2":
    #     sample_text['words']=sample_text['text'].str.split()
    # elif args.dataset_type == "tsd":
    #     sample_text['words']=sample_text['text'].str.split(' ')
    sample_text['words']=sample_text['text'].str.split(' ')
    sample_text['words_index']=sample_text['words'].apply(lambda x: list(range(len(x))))

    sample_text['tok2char']=sample_text.apply(lambda x: get_tok2char(x, args.dataset_type),axis=1)
    sample_text['word2char']=sample_text.apply(lambda x: get_word2char(x, " "),axis=1)
    # if args.dataset_type == "gpt2":
    #     sample_text['word2tok']=sample_text.apply(lambda x: get_word2tok(x),axis=1)
    # elif args.dataset_type == "tsd":
    #     sample_text['word2tok']=sample_text.apply(lambda x: get_word2tok(x, " "),axis=1)
    sample_text['word2tok']=sample_text.apply(lambda x: get_word2tok(x, " "),axis=1)
    sample_text['tok2word']=sample_text['word2tok'].apply(kv_swap)
    sample_text['char2tok']=sample_text['tok2char'].apply(kv_swap)
    sample_text['char2word']=sample_text['word2char'].apply(kv_swap)

    ## predictions_labels에 다시 merge
    predictions_labels = pd.merge(predictions_labels, sample_text[['text','words','char','tok2char', 'word2char', 'word2tok','tok2word', 'char2tok', 'char2word']],on='text',how='left')

    ## convert list of indices into a list of binary labels of length len(seq)        
    predictions_labels['pred_binary']=predictions_labels.apply(lambda x: index2binary(x, len_colname="tokens", index_colname="pred"),axis=1)

    predictions_labels['pred_word']=predictions_labels.apply(get_pred_word,axis=1)
    predictions_labels['pred_binary_word']=predictions_labels.apply(get_pred_binary_word,axis=1)
    predictions_labels['pred_scores_word']=predictions_labels.apply(lambda x: get_pred_scores_word(x,method='max'), axis=1) 

    predictions_labels['pred_char']=predictions_labels.apply(get_pred_char,axis=1)
    predictions_labels['pred_binary_char']=predictions_labels.apply(get_pred_binary_char,axis=1)
    predictions_labels['pred_scores_char']=predictions_labels.apply(lambda x: get_pred_scores_char(x), axis=1) 

    if args.dataset_type == "gpt2":
        ## convert list of indices into a list of binary labels of length len(seq)        
        predictions_labels['labels_binary'] = predictions_labels['labels'].apply(lambda x: [1 if i >= 0.5 else 0 for i in x])
        predictions_labels['labels_binary_word']=predictions_labels.apply(lambda x: get_labels_binary_word(x, dataset_type=args.dataset_type),axis=1)
        predictions_labels['labels_binary_char']=predictions_labels.apply(get_labels_binary_char,axis=1)
    elif args.dataset_type == "tsd":
        predictions_labels['labels_binary_char'] = predictions_labels.apply(lambda x: index2binary(x, len_colname="char", index_colname="labels_index_char"),axis=1)
        predictions_labels['labels_binary_word']=predictions_labels.apply(lambda x: get_labels_binary_word(x, dataset_type=args.dataset_type),axis=1)
        predictions_labels['labels_binary']=predictions_labels.apply(lambda x: get_labels_binary_token(x, dataset_type=args.dataset_type),axis=1)

    ## print stats of datasets
    data_stats_f.write(f"# Samples: {predictions_labels.shape[0]}\n")
    data_stats_f.write(f"Avg. # Tokens per Sample: {predictions_labels['tokens'].apply(len).mean()}\n")
    data_stats_f.write(f"Avg. # Tokens Located per Sample: {predictions_labels['labels_binary'].apply(sum).mean()}\n")
    data_stats_f.write(f"Avg. # Characters per Sample: {predictions_labels['char'].apply(len).mean()}\n")
    data_stats_f.write(f"Avg. # Characters Located per Sample: {predictions_labels['labels_binary_char'].apply(sum).mean()}\n")
    data_stats_f.write(f"Avg. # Words per Sample: {predictions_labels['words'].apply(len).mean()}\n")
    data_stats_f.write(f"Avg. # Words Located per Sample: {predictions_labels['labels_binary_word'].apply(sum).mean()}\n")
    data_stats_f.close()

    # ### Calculate Token-level Metrics
    predictions_labels['f1']=predictions_labels.apply(apply_f1,axis=1)
    predictions_labels['f2']=predictions_labels.apply(apply_f2,axis=1)
    predictions_labels['rr']=predictions_labels.apply(get_rr, axis=1)
    predictions_labels['ap']=predictions_labels.apply(apply_ap,axis=1)
    predictions_labels['precision']=predictions_labels.apply(apply_precision,axis=1)
    predictions_labels['recall']=predictions_labels.apply(apply_recall,axis=1)

    ## Summary metric
    ## ToDo : double check if mean F1, mean F2 is a thing. -> Not sure.. Ask Jong?
    ## double checked what happens if true label is none. ap -> np.nan, rr -> np.nan, f1 -> np.nan, f2 -> np.nan
    
    # ### Calculate Word-level Metrics
    predictions_labels['f1_word']=predictions_labels.apply(lambda x: apply_f1(x,"_word"),axis=1)
    predictions_labels['f2_word']=predictions_labels.apply(lambda x: apply_f2(x,"_word"),axis=1)
    predictions_labels['ap_word']=predictions_labels.apply(lambda x: get_rr(x,"_word"),axis=1)
    predictions_labels['rr_word']=predictions_labels.apply(lambda x: apply_ap(x,"_word"),axis=1)
    predictions_labels['precision_word']=predictions_labels.apply(lambda x: apply_precision(x,"_word"),axis=1)
    predictions_labels['recall_word']=predictions_labels.apply(lambda x: apply_recall(x,"_word"),axis=1)

    ## Summary metric
    mf1 = predictions_labels['f1_word'].mean()
    mf2 = predictions_labels['f2_word'].mean()
    mrr = predictions_labels['rr_word'].mean()
    map_score =  predictions_labels['ap_word'].mean()
    precision = predictions_labels['precision_word'].mean()
    recall = predictions_labels['recall_word'].mean()

    summary_result.update({"mf1_word": [mf1],
                           "mf2_word": [mf2],
                           "mrr_word": [mrr],
                           "map_word": [map_score],
                           "precision_word": [precision], 
                           "recall_word": [recall]})
    
    mf1 = predictions_labels['f1'].mean()
    mf2 = predictions_labels['f2'].mean()
    mrr = predictions_labels['rr'].mean()
    map_score =  predictions_labels['ap'].mean()
    precision = predictions_labels['precision'].mean()
    recall = predictions_labels['recall'].mean()

    summary_result.update({"mf1_token": [mf1],
                           "mf2_token": [mf2],
                           "mrr_token": [mrr],
                           "map_token": [map_score],
                           "precision_token": [precision], 
                           "recall_token": [recall]})

    # ### Calculate Character-level Metrics
    predictions_labels['f1_char']=predictions_labels.apply(lambda x: apply_f1(x,"_char"),axis=1)
    predictions_labels['f2_char']=predictions_labels.apply(lambda x: apply_f2(x,"_char"),axis=1)
    predictions_labels['ap_char']=predictions_labels.apply(lambda x: get_rr(x,"_char"),axis=1)
    predictions_labels['rr_char']=predictions_labels.apply(lambda x: apply_ap(x,"_char"),axis=1)
    predictions_labels['precision_char']=predictions_labels.apply(lambda x: apply_precision(x,"_char"),axis=1)
    predictions_labels['recall_char']=predictions_labels.apply(lambda x: apply_recall(x,"_char"),axis=1)

    ## Summary metric
    mf1 = predictions_labels['f1_char'].mean()
    mf2 = predictions_labels['f2_char'].mean()
    mrr = predictions_labels['rr_char'].mean()
    map_score =  predictions_labels['ap_char'].mean()
    precision = predictions_labels['precision_char'].mean()
    recall = predictions_labels['recall_char'].mean()

    summary_result.update({"mf1_char": [mf1],
                           "mf2_char": [mf2],
                           "mrr_char": [mrr],
                           "map_char": [map_score],
                           "precision_char": [precision], 
                           "recall_char": [recall]})

    
    # predictions_labels.to_json("new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_gn_metrics.jsonl", lines=True, orient='records')
    predictions_labels.to_excel("_".join([os.path.splitext(args.pred_file_path)[0],"metrics.xlsx"]), index=False)
    pd.DataFrame(summary_result).to_csv("_".join([os.path.splitext(args.pred_file_path)[0],"metrics_summary.csv"]), index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file_path", type=str, default="new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_gn.jsonl", required=True)
    parser.add_argument("--label_file_path", type=str, default="new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl", required=True)
    parser.add_argument("--method", type=str, choices=["attention", "grad_norm"], required=True)
    parser.add_argument("--dataset_type", type=str, choices=["gpt2", "tsd"], required=True)
    parser.add_argument("--tokenizer_path", type=str, default="/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-2/step_2600_best_checkpoint/", required=True)
    

    args = parser.parse_args()
    print(type(args))
    #### Data Specific Code ####
    # ### Prepare dataset (predictions & labels)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    if args.dataset_type == "gpt2":
        ## read predicted file
        pred_path = args.pred_file_path #"new_module/locate/results/toxicity/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/testset_gpt2_2500_gn.jsonl"
        predictions = pd.read_json(pred_path, lines=True)
        predictions = predictions[['prompt','text',f'pred_indices_{args.method}', f'pred_scores_{args.method}']].copy()
        predictions = predictions.rename(columns={f'pred_indices_{args.method}':'pred',
                                                f'pred_scores_{args.method}':'pred_scores'})

        ## clean text column -> remove "<|endoftext|>" text
        predictions['text']=predictions['text'].str.replace("<|endoftext|>","")

        ## read ground truth file
        label_path = args.label_file_path #"new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl"
        labels = pd.read_json(label_path, lines=True)

        ## unravel the file 
        labels['prompt']=labels['prompt'].apply(lambda x: x['text'])

        labels = labels.explode('generations')

        labels['text']=labels['generations'].apply(lambda x: x['text'])
        labels['tokens']=labels['generations'].apply(lambda x: x['tokens'])

        labels['locate_labels']=labels['generations'].apply(lambda x: x.get('locate_labels', np.nan))

        del labels['generations']

        labels = labels.rename(columns={'locate_labels':'labels'})
        labels = labels.dropna(subset='labels')

        ## correct minor errors -> remove trailing pad_token in the generations.
        has_pad_token = labels['tokens'].apply(count_pad_token) > 0
        labels.loc[has_pad_token, 'labels'] = labels.loc[has_pad_token,:].apply(lambda x: remove_label_for_pad_token(x, 'labels'),axis=1).values
        labels.loc[has_pad_token, 'tokens'] = labels.loc[has_pad_token, 'tokens'].apply(remove_pad_token)

        ## similarly, clean text column -> remove "<|endoftext|>" text
        labels['text']=labels['text'].str.replace("<|endoftext|>","")

        #### COMMON CODE 
        ## join predictions & labels 
        predictions = pd.merge(predictions, labels, on=['prompt','text'],how='left')
        ## drop duplicates (cases where different labels for the same text)
        predictions = predictions.drop_duplicates(subset=['prompt','text'],keep=False)
        ## double check that there's no rows with mismatched length of labels and predictions
        assert len(predictions.loc[predictions['labels'].apply(len) != predictions['pred_scores'].apply(len), :])==0

    elif args.dataset_type == "tsd":
        ## read predicted file
        pred_path = args.pred_file_path
        predictions = pd.read_json(pred_path, lines=True)
        predictions = predictions[['text',f'pred_indices_{args.method}', f'pred_scores_{args.method}']].copy()
        predictions = predictions.rename(columns={f'pred_indices_{args.method}':'pred',
                                                f'pred_scores_{args.method}':'pred_scores'})
        
        ## read ground truth file
        label_path = args.label_file_path #"new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl"
        labels = pd.read_json(label_path, lines=True)
        labels = labels[['text','spans']].copy()
        labels = labels.rename(columns={'spans':'labels_index_char'})
        labels['labels_index_char'] = labels['labels_index_char'].apply(eval)
        
        ## join predictions & labels 
        predictions = pd.merge(predictions, labels, on=['text'],how='left')
        ## drop duplicates (cases where different labels for the same text)
        predictions = predictions.drop_duplicates(subset=['text'],keep=False)
        predictions = predictions.loc[predictions['labels_index_char'].apply(len) > 0, :].copy()
        

    calculate_locate_metrics(predictions, args)