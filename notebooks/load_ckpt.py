import torch
import os
import json

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AddedToken
from torch import optim

import numpy as np

params = ['', 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification',
 '0,1',
 'train',
 'dev',
 'test',
 'roberta-base',
 'models_bak_contd/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds',
 'gpt2-roberta',
 'full',
 'gpt2-large',
 'freeze-vecmap',
 'dontbinarize',
 'jsonl']


# config 

def define_model(mod_path):
    
    base_path = params[1]
    binarize_labels = False
    if len(params) > 12:
        binarize_labels = params[12] == "binarize_labels"

    filetype = "txt"
    if len(params) > 13:
        filetype = params[13]

    labels = [int(label) for label in params[2].split(",")]

    tokenizer_ = AutoTokenizer.from_pretrained(params[6], cache_dir="hf_cache")
    if params[10] != "none":
        tokenizer = AutoTokenizer.from_pretrained(params[10], cache_dir="hf_cache")
        tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)
    else:
        tokenizer = tokenizer_
        # tokenizer = AutoTokenizer.from_pretrained(params[6], cache_dir="hf_cache")
        
    config = AutoConfig.from_pretrained(params[6], cache_dur="hf_cache", num_labels=len(labels))
    config2 = None
    if params[10] != "none":  
        config2 = AutoConfig.from_pretrained(params[10], cache_dur="hf_cache", num_labels=len(labels))
        print(config2.pad_token_id)
        config2.pad_token_id = tokenizer.pad_token_id
        print(config2.pad_token_id)
        print("look above for padding")

        tokenizer_ = AutoTokenizer.from_pretrained(params[6], config=config)
        tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)

    if params[8] == "gpt2-roberta":
        SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
        # config.pad_token_id = tokenizer.eos_token_id
        print("Adding special tokens")
        tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # if params[10] != "none":
    model = AutoModelForSequenceClassification.from_pretrained(params[10], config=config2) # unindented
    # model.resize_token_embeddings(len(tokenizer))

    def learn_vecmap(X, y):
        print("computing vecmap")
        w = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(y)
        vecmap = torch.nn.Linear(w.size(0), w.size(1), bias=False)
        print(w.size(), vecmap.weight.size())
        vecmap.weight.data.copy_(w.data.t())
        return vecmap

    def vocab_permutation(vocab1, vocab2):
        vocab2itos = {k:v for v,k in vocab2.items()}
        vocab2list = [vocab2itos[k] for k in range(len(vocab2itos))]

        perm1 = []
        perm2 = []
        unincluded = []
        for i, word in enumerate(vocab2list):
            if word in vocab1:
                perm1.append(vocab1[word])
                perm2.append(i)
            else:
                unincluded.append(word)

        print(unincluded)
        return perm1, perm2

    embeds = model.get_input_embeddings()
    new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
    for p in new_embeds.parameters():
        p.requires_grad = False

    new_embeds.weight.data.copy_(embeds.weight)
    config.new_n_embd = new_embeds.embedding_dim
    config.new_vocab_size = new_embeds.num_embeddings

    model_ = AutoModelForSequenceClassification.from_pretrained(params[6], config=config)
    tokenizer_ = AutoTokenizer.from_pretrained(params[6], config=config)

    perm, perm_ = vocab_permutation(tokenizer.vocab, tokenizer_.vocab)
    old_embeds = model_.get_input_embeddings()
    vecmap = learn_vecmap(new_embeds.weight[perm], old_embeds.weight[perm_])
    new_embeds = torch.nn.Sequential(new_embeds, vecmap)
    model_.set_input_embeddings(new_embeds)
    model = model_


    # state_dict
    mod = torch.load(mod_path)
    model.load_state_dict(mod)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    return model, config, tokenizer
    

if __name__ == "__main__":
    
    pass