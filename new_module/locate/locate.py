# -*- coding: utf-8 -*-
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
project_dir=os.path.join(os.path.join(os.path.dirname(__file__),os.pardir),os.pardir)
os.chdir(project_dir)
sys.path.append(project_dir)
import string
import argparse

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np

from new_module.utils.robertacustom import RobertaCustomForSequenceClassification

def locate_attn(attentions, tokenizer, batch, max_num_tokens = 6, num_layer=10, unit="word", use_cuda=True):

    punctuations = string.punctuation + '\n '
    punctuations = list(punctuations)
    punctuations.remove('-')

    ## attentions : tuple of length num hidden layers
    ## attentions[i] : attention value of ith hidden layer of shape (batch, num_heads, query, value)
    lengths = [i.tolist().count(1) for i in batch["attention_mask"]]
    print(lengths[1])
    # 보고자 하는 attention layer 만 가져옴
    attentions = attentions[
        num_layer # originally 10
    ]
    print(attentions.shape)
    print(attentions.max(1)[0].shape)
    print( batch["input_ids"].shape)
    print( batch["attention_mask"][0,:])
    cls_attns = attentions.max(1)[0][:, 0]
    
    stopwords = [" and", " of", " or", " so"] + punctuations + [token for token in tokenizer.special_tokens_map.values()]
    stopwords_ids = [tokenizer.encode(word,add_special_tokens=False)[-1] for word in stopwords]
    # print("stopwords_ids", torch.tensor(stopwords_ids))

    locate_ixes=[]
    for i, attn in enumerate(cls_attns):
        
        print("attn.shape", attn.shape)
        current_sent = batch["input_ids"][i][: lengths[i]]
        print("current_sent", current_sent)
        if use_cuda:
            no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids).to(torch.device('cuda'))))[0]
        else:
            no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids)))[0]
        print("no_punc_indices", no_punc_indices)
        print(f"current_sent[no_punc_indices]: {current_sent[no_punc_indices]}")
        print(f"tokenizer.decode(current_sent[no_punc_indices]): {tokenizer.decode(current_sent[no_punc_indices])}")
        
        # current tokenizer does not add <s> and </s> to the sentence.
        current_attn = attn[: lengths[i]].softmax(-1) 
        # print("current_attn", current_attn)
        current_attn = current_attn[no_punc_indices]
        # print("current_attn", current_attn)
        
        # 이 값의 평균을 구한다.
        avg_value = current_attn.view(-1).mean().item()
        # print("avg_value", avg_value)
        # 이 값 중에 평균보다 큰 값을 지니는 위치를 찾는다.
        # fixed to reflect that sometimes the sequence length is 1.
        top_masks = ((current_attn >= avg_value).nonzero().view(-1)) 
        torch.cuda.empty_cache()
        top_masks = top_masks.cpu().tolist()
        print("top_masks", top_masks)
        
        # attention 값이 평균보다 큰 토큰의 수가 k개 또는 문장 전체 토큰 수의 1/3 보다 크면  
        if len(top_masks) > min((lengths[i]) // 3, max_num_tokens):
            # 그냥 attention 값 기준 k 개 또는 토큰 수/3 중 작은 수를 뽑는다.
            top_masks = (
                current_attn.topk(max(min((lengths[i]) // 3, max_num_tokens), 1))[1]
            )
            top_masks = top_masks.cpu().tolist()
            # print("top k top_masks", top_masks)
        top_masks_final = no_punc_indices[top_masks]
        # print("top_masks_final", top_masks_final)
        if unit == "token":
            locate_ixes.append(list(set(top_masks_final.cpu().detach().tolist())))
        
        elif unit == "word":
            # word의 일부만 locate 한 경우, word 전체를 locate 한다.
            # 같은 word 안에 있는 token 끼리 묶음.
            j, k = 0, 0
            grouped_tokens = []
            grouped_tokens_for_word = []
            words = tokenizer.decode(current_sent).strip().split()
            # print("words", words)
            while j < len(current_sent):
                if (tokenizer.decode(current_sent[j]).strip() not in stopwords):
                    # print("tokenizer.decode(current_sent[j])", tokenizer.decode(current_sent[j]))
                    while k < len(words):
                        if tokenizer.decode(current_sent[j]).strip() in words[k]:
                            grouped_tokens_for_word.append(j)
                            break
                        else:
                            grouped_tokens.append(grouped_tokens_for_word)
                            grouped_tokens_for_word = []
                            k += 1
                j += 1
            grouped_tokens.append(grouped_tokens_for_word)
            # print("grouped_tokens", grouped_tokens)
            
            top_masks_final.sort()
            top_masks_final_final = []
            for index in top_masks_final:
                # print("index", index)
                if index not in top_masks_final_final:
                    word = [grouped_ixes for grouped_ixes in grouped_tokens if index in grouped_ixes]
                    # print("word", word)
                    if len(word) > 0:
                        word = word[0]
                    else:
                        print(f"!!! {index} not in the grouped_ixes {grouped_tokens}")
                        print(f"!!! tokenizer.decode(index): {tokenizer.decode(index)}")
                    top_masks_final_final.extend(word)
            locate_ixes.append(list(set(top_masks_final_final)))

            
    return locate_ixes


def locate_grad_norm(output, tokenizer, batch, label_id = 1, max_num_tokens = 6, unit="word", use_cuda=True):

    punctuations = string.punctuation + '\n '
    punctuations = list(punctuations)
    punctuations.remove('-')
    stopwords = [" and", " of", " or", " so"] + punctuations + [token for token in tokenizer.special_tokens_map.values()]
    stopwords_ids = [tokenizer.encode(word,add_special_tokens=False)[-1] for word in stopwords]

    ## output['hidden_states']: tuple of length num_hidden_layers
    ## output['hidden_states'][0]: (batch_size, seq_len, hidden_size)
    layer = output['hidden_states'][0]
    layer.retain_grad()
    
    ## output['logits'] : (batch_size, num_labels)
    softmax=torch.nn.Softmax(dim=-1)
    probs = softmax(output['logits'])[:, label_id]
    # print(f"probs.shape:{probs.shape}")
    
    probs.sum().backward(retain_graph=True)

    ## layer.grad : (batch_size, seq_len, hidden_size)
    # print(f"layer.grad.shape:{layer.grad.shape}")
    norm = torch.norm(layer.grad, dim=-1)
    ## norm : (batch_size, seq_len)
    # print(f"norm.shape:{norm.shape}")
    norm = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10))
    # print(f"norm:{norm}")
    
    lengths = [i.tolist().count(1) for i in batch["attention_mask"]]
    # print(f"lengths: {lengths}")
    
    locate_ixes = []
    locate_scores = []
    for i in range(batch["input_ids"].shape[0]):
        
        ## norm_ : (seq_len,)
        current_norm = norm[i, :]
        # print(f"norm_ shape: {current_norm.shape}")
        
        ## current_sent : (lengths[i], )
        current_sent = batch["input_ids"][i][: lengths[i]]
        if use_cuda:
            no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids).to(torch.device('cuda'))))[0]
        else:
            no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids)))[0]
        # print(f"current_sent: {current_sent}")
        # print(f"len(current_sent), lengths[i]: {len(current_sent), lengths[i]}")
        # print(f"no_punc_indices: {no_punc_indices}")
        # print(f"current_sent[no_punc_indices]: {current_sent[no_punc_indices]}")
        # print(f"tokenizer.decode(current_sent[no_punc_indices]): {tokenizer.decode(current_sent[no_punc_indices])}")
        
        ## normalize current_norm
        current_norm = current_norm[: lengths[i]].softmax(-1) 
        # print(f"current_norm after normalizing: {current_norm}")
        
        current_locate_scores = torch.zeros_like(current_norm)
        current_locate_scores[no_punc_indices] = current_norm[no_punc_indices].clone()
        locate_scores.append(current_locate_scores.cpu().detach().tolist())
        
        current_norm = current_norm[no_punc_indices]
        # print(f"current_norm after selecting non stop words indices: {current_norm}")
        
        ## calculate mean value within the sequence
        avg_value = current_norm.view(-1).mean().item()
        # print("avg_value", avg_value)
        
        ## find indices of tokens whose norm value is greater than the mean value
        top_masks = ((current_norm >= avg_value).nonzero().view(-1)) 
        top_masks = top_masks.cpu().tolist()
        torch.cuda.empty_cache()
        # print("indices of non stopwords tokens whose grad norm value is greater than the mean value", top_masks)
        
        ## in case the number of above average gradient norm tokens is greater than the max_num_tokens or 1/3 of the lengths[i]
        if len(top_masks) > min((lengths[i]) // 3, max_num_tokens):
            # print("len(located_tokens) exceeds max_num_tokens or 1/3 of the lengths[i]. Taking top k.")
            top_masks = (
                current_norm.topk(max(min((lengths[i]) // 3, max_num_tokens), 1))[1]
            )
            top_masks = top_masks.cpu().tolist()
            # print("indices of non stopwords tokens located after taking top k", top_masks)
        
        top_masks = no_punc_indices[top_masks].cpu().detach().tolist()
        # print("indices of tokens located", top_masks)
        
        if unit == "token":
            locate_ixes.append(list(set(top_masks)))
        
        elif unit == "word":

            ## group token indices that belong to the same word
            j, k = 0, 0
            grouped_tokens = []
            grouped_tokens_for_word = []
            words = tokenizer.decode(current_sent).strip().split()
            while j < len(current_sent):
                if (tokenizer.decode(current_sent[j]).strip() not in stopwords):
                    while k < len(words):
                        if tokenizer.decode(current_sent[j]).strip() in words[k]:
                            grouped_tokens_for_word.append(j)
                            break
                        else:
                            grouped_tokens.append(grouped_tokens_for_word)
                            grouped_tokens_for_word = []
                            k += 1
                j += 1
            grouped_tokens.append(grouped_tokens_for_word)
            
            ## expand located token indices to include adjacent token indices that belong to the same word as already located tokens
            top_masks.sort()
            top_masks_final = set()
            for index in top_masks:
                if index not in top_masks_final:
                    word = [grouped_ixes for grouped_ixes in grouped_tokens if index in grouped_ixes]
                    # print("word", word)
                    if len(word) > 0:
                        word = set(word[0])
                    else:
                        print(f"warning. {index} not in the word groups. decoded value: {tokenizer.decode(index)}")
                        word = set([index])
                    top_masks_final |= word
            locate_ixes.append(sorted(list(top_masks_final)))

            
    return locate_ixes, locate_scores

def locate_main(method, model, tokenizer, batch, max_num_tokens = 6, num_layer=10, unit="word", use_cuda=True):
    
    if method == "attn":
        output = model(**batch, output_attentions=True)
        return locate_attn(output.attentions, tokenizer, batch, max_num_tokens = 6, num_layer=10, unit="word", use_cuda=True)
    elif method == "grad_norm":
        output = model(**batch, output_hidden_states=True)
        return locate_grad_norm(output, tokenizer, batch, max_num_tokens = 6, unit="word", use_cuda=True)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""Program to locate spans that contributed the most to the prediction of a model\n\
Input format: jsonl or csv with a column named "text" containing the text to be analyzed\n\
Output format: input dataframe with a new column named "located_indices" each of which is a list of indices of tokens. e.g. [2,3,4,8,10]\n\
""")
    parser.add_argument("--method", type=str, help="method to use for locating tokens to edit")
    parser.add_argument("--input_path", type=str, help="path to input file")
    parser.add_argument("--output_path", type=str, help="path to output file")
    parser.add_argument("--model_name_or_path", type=str, help="name of model to use or path to the checkpoint to use")
    parser.add_argument("--model_type", type=str, choices=["AutoModelForSequenceClassification", "RobertaCustomForSequenceClassification"], help="name of model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to use")
    parser.add_argument("--text_column_name", type=str, default="text", help="name of the column containing text for analysis")
    
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_type == "AutoModelForSequenceClassification":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    elif args.model_type == "RobertaCustomForSequenceClassification":
        model = RobertaCustomForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.to(device)
        
    if args.input_path.endswith(".jsonl"):
        data = pd.read_json(args.input_path, lines=True)
    elif args.input_path.endswith(".csv"):
        data = pd.read_csv(args.input_path)
        
    if "gpt2" in args.input_path: ## unravel data
        print("GPT2 in args.input_path")
        ## unravel the file 
        data['prompt']=data['prompt'].apply(lambda x: x['text'])
        data = data.explode('generations')

        data['text']=data['generations'].apply(lambda x: x['text'])
        data['tokens']=data['generations'].apply(lambda x: x['tokens'])
        data['locate_labels']=data['generations'].apply(lambda x: x.get('locate_labels', np.nan))
        data = data.dropna(subset=['locate_labels'])
        
        del data['generations']
        del data['locate_labels']
        print(data.head())
    
    dataset = Dataset.from_pandas(data)
    if "gpt2" in args.input_path:
        print("GPT2 in args.input_path")
        def collate_fn(batch):
            input_ids = pad_sequence([torch.LongTensor(example["tokens"]) for example in batch], padding_value=tokenizer.pad_token_id, batch_first=True) 
            print(f"input_ids: {input_ids}")
            batch = {"input_ids": input_ids,
                    "attention_mask": (input_ids != tokenizer.pad_token_id).long()}
            return transformers.tokenization_utils_base.BatchEncoding(batch)
    else:
        def collate_fn(batch):
            batch = tokenizer([example[args.text_column_name] for example in batch], padding=True, truncation=True, return_tensors="pt")
            return batch
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    pred_indices = []
    pred_scores = []
    for batch in dataloader:
        batch.to(device)
        results, scores = locate_main(args.method, model, tokenizer, batch, max_num_tokens = 6, num_layer=10, unit="word", use_cuda=True)
        pred_indices.extend(results)
        pred_scores.extend(scores)
    
    data[f'pred_indices_{args.method}'] = pred_indices
    data[f'pred_scores_{args.method}'] = pred_scores
    data.to_json(args.output_path, lines=True, orient='records')