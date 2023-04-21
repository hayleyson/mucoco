# -*- coding: utf-8 -*-

# attempt at resuming from a terminated run (originally trained with transformers v 4.2.1)
# problems 1) Trainer.train(resume_from_checkpoint = xx) only came out after v4.2.1 (ver. I originally used.)
#            So, I updated the transformers to .. v4.26.1.. Behaviors might be slightly different between two versions.
#          2) In the github, there's statement that you need to install transformers >= 4.5.1. 
#            But I originally used v4.2.1. The training might not replicate Sachin's experiments.
# --> 2 is critical. 
# ----> Should I retrain the model from scratch?
# ----> If so, better write up a continuous training script and use it...T.T

# #train the classifier
# python -u examples/training_constraint_models/train_classifier.py\
#     data/toxicity/jigsaw-unintended-bias-in-toxicity-classification\
#     0,1\
#     train\
#     dev\
#     test\
#     roberta-base\
#     models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds\
#     gpt2-roberta full gpt2-large freeze-vecmap dontbinarize jsonl

# define arguments
params = ['', 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification',
 '0,1',
 'train',
 'dev',
 'test',
 'roberta-base',
 'models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds',
 'gpt2-roberta',
 'full',
 'gpt2-large',
 'freeze-vecmap',
 'dontbinarize',
 'jsonl']

ckpt_dir_par='/home/hyeryungson/mucoco/models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds'
###############################################################################################################
pjt_id = 'gvho0s98'
###############################################################################################################

import wandb
import torch
import os
import json
###############################################################################################################
from glob import glob 
import yaml
num_gpu = torch.cuda.device_count()

# for resume in wandb
wandb.init(project="huggingface", resume="must", id=pjt_id)
wandb_path=sorted(glob(f'/home/hyeryungson/mucoco/wandb/run-*{pjt_id}'), reverse=True)[0]
config_path=os.path.join(wandb_path, 'files/config.yaml')
print('path to yaml file', config_path)
with open(config_path, 'r') as stream:
    past_run_config = yaml.safe_load(stream)
    
# set the ckpt with highest step number as ckpt path
ckpt_dir=sorted(glob(f'{ckpt_dir_par}/results/*/'), reverse=True)[0]
print(ckpt_dir)
###############################################################################################################

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AddedToken
from torch import optim

import numpy as np
###############################################################################################################
from .load_ckpt import define_model
###############################################################################################################
# os.makeirs(params[7], exist_ok=True)

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

base_path = params[1]
binarize_labels = False
if len(params) > 12:
    binarize_labels = params[12] == "binarize_labels"

filetype = "txt"
if len(params) > 13:
    filetype = params[13]

labels = [int(label) for label in params[2].split(",")]
train_paths = []
valid_paths = []
test_paths = []
for label in labels:
    train_paths.append(open(f"{base_path}/{params[3]}_{label}.{filetype}"))
    valid_paths.append(open(f"{base_path}/{params[4]}_{label}.{filetype}"))
    test_paths.append(open(f"{base_path}/{params[5]}_{label}.{filetype}"))

def create_dataset(paths, labelses):
    texts, labels = [], []
    # print(paths)
    for i, path in enumerate(paths):
        for l in path:
            if filetype == "jsonl":
                text = json.loads(l)["text"]
            else:
                text = l.strip()
            if binarize_labels:
                label = 0
                if labelses[i] <= 2:
                    label = 0
                    texts.append(text)
                    labels.append(label)
                elif labelses[i] >= 3:
                    label = 1
                    texts.append(text)
                    labels.append(label)
            else:
                labels.append(labelses[i])
                texts.append(text)
            
    print("create_dataset", len(texts), len(labels), set(labels))
    return texts, labels
    
train_texts, train_labels = create_dataset(train_paths, labels)
val_texts, val_labels = create_dataset(valid_paths, labels)
test_texts, test_labels = create_dataset(test_paths, labels)

# tokenizer = AutoTokenizer.from_pretrained(params[6], cache_dir="hf_cache")
# config = AutoConfig.from_pretrained(params[6], cache_dur="hf_cache", num_labels=len(labels))

###############################################################################################################
# tokenizer_ = AutoTokenizer.from_pretrained(params[6], cache_dir="hf_cache")
# if params[10] != "none":
#     tokenizer = AutoTokenizer.from_pretrained(params[10], cache_dir="hf_cache")
#     tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)
# else:
#     tokenizer = tokenizer_
#     # tokenizer = AutoTokenizer.from_pretrained(params[6], cache_dir="hf_cache")

# config = AutoConfig.from_pretrained(params[6], cache_dur="hf_cache", num_labels=len(labels))
# config2 = None
# if params[10] != "none":  
#     config2 = AutoConfig.from_pretrained(params[10], cache_dur="hf_cache", num_labels=len(labels))
#     print(config2.pad_token_id)
#     config2.pad_token_id = tokenizer.pad_token_id
#     print(config2.pad_token_id)
#     print("look above for padding")

#     tokenizer_ = AutoTokenizer.from_pretrained(params[6], config=config)
#     tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)
###############################################################################################################


# config.n_positions = max_length
# config.max_position_embeddings = max_length
# if params[8] == "krishna":
#     SPECIAL_TOKENS = {
#         "additional_special_tokens": ["<dense-vectors>", "<tokens>", "<verb>", "<ARG0>", "<ARG1>", "<global-dense-vectors>"],
#         "pad_token": "<eos>",
#         "bos_token": "<bos>",
#         "eos_token": "<eos>"
#     }
#     print("Adding special tokens")
#     tokenizer.add_special_tokens(SPECIAL_TOKENS)
#     config.pad_token_id = tokenizer.pad_token_id

# elif params[8] == "roberta" :
#     SPECIAL_TOKENS = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
#     print("Adding special tokens")
#     tokenizer.add_special_tokens(SPECIAL_TOKENS)
#     config.pad_token_id = tokenizer.pad_token_id

# elif params[8] == "dialogpt": 
#     SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
#     print("Adding special tokens")
#     tokenizer.add_special_tokens(SPECIAL_TOKENS)
#     config.pad_token_id = tokenizer.pad_token_id

# elif params[8] == "gpt2":
#     SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
#     config.pad_token_id = tokenizer.eos_token_id
#     # tokenizer.pad_token_id = tokenizer.eos_token_id
#     print("Adding special tokens")
#     tokenizer.add_special_tokens(SPECIAL_TOKENS)
#     print(tokenizer)

# elif params[8] == "bert":
#     # SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
#     # config.pad_token_id = tokenizer.eos_token_id
#     # print("Adding special tokens")
#     # tokenizer.add_special_tokens(SPECIAL_TOKENS)
#     pass

# elif params[8] == "gpt2-roberta":
###############################################################################################################
# if params[8] == "gpt2-roberta":
#     SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
#     # config.pad_token_id = tokenizer.eos_token_id
#     print("Adding special tokens")
#     tokenizer.add_special_tokens(SPECIAL_TOKENS)

# # elif params[8] == "gpt2-distilbert":
# #     SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
# #     # config.pad_token_id = tokenizer.eos_token_id
# #     print("Adding special tokens")
# #     tokenizer.add_special_tokens(SPECIAL_TOKENS)


# tokenizer.save_pretrained(f"{params[7]}/checkpoint_best")
###############################################################################################################

###############################################################################################################
# ckpt_path='/home/hyeryungson/mucoco/models_bak_contd/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best/pytorch_model.bin'
model, config, tokenizer = define_model(os.path.join(ckpt_dir, "pytorch_model.bin"))
###############################################################################################################


# if params[9] != "only_tokenizer": # all below are inside this if statement

print("tokenizer loaded")
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
print(test_encodings['input_ids'][0], len(test_encodings['input_ids'][0]))
# input()
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)

print("datasets loaded and tokenizer")

# if params[10] != "none":
###############################################################################################################
# model = AutoModelForSequenceClassification.from_pretrained(params[10], config=config2) # unindented
###############################################################################################################

# model.resize_token_embeddings(len(tokenizer))

#         if params[11] == "random":  
#             embeds = model.get_input_embeddings()
#             embeds.weight.data.normal_(mean=0.0, std=0.02)
#             if embeds.padding_idx is not None:
#                 embeds.weight.data[module.padding_idx].zero_()
#             model.resize_token_embeddings(len(tokenizer))

#         elif params[11] == "freeze":
#             model = AutoModelForSequenceClassification.from_pretrained(params[6], config=config)
#             embeds = model.get_input_embeddings()
#             for p in embeds.parameters():
#                 p.requires_grad = False
#             # embeds.requires_grad=False
#             model.resize_token_embeddings(len(tokenizer))

#         elif params[11] == "freeze-project":
#             embeds = model.get_input_embeddings()
#             new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
#             for p in new_embeds.parameters():
#                 p.requires_grad = False
#             # new_embeds.requires_grad = False
#             new_embeds.weight.data.copy_(embeds.weight)
#             print(model.device)
#             config.new_n_embd = new_embeds.embedding_dim
#             config.new_vocab_size = new_embeds.num_embeddings
#             # if params[8] == "gpt2-roberta":
#             #     config.pad_token_id = tokenizer.eos_token_id
#             model_ = AutoModelForSequenceClassification.from_pretrained(params[6], config=config)
#             new_embeds = torch.nn.Sequential(new_embeds, torch.nn.Linear(new_embeds.embedding_dim, model_.get_input_embeddings().embedding_dim, bias=False))
#             model_.set_input_embeddings(new_embeds)
#             model = model_

#         elif params[11] == "freeze-eye":
#             embeds = model.get_input_embeddings()
#             new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
#             for p in new_embeds.parameters():
#                 p.requires_grad = False
#             # new_embeds.requires_grad = False
#             new_embeds.weight.data.copy_(embeds.weight)
#             print(model.device)
#             config.new_n_embd = new_embeds.embedding_dim
#             # if params[8] == "gpt2-roberta":
#             #     config.pad_token_id = tokenizer.eos_token_id
#             model_ = AutoModelForSequenceClassification.from_pretrained(params[6], config=config)
#             eye = torch.nn.Linear(new_embeds.embedding_dim, model_.get_input_embeddings().embedding_dim, bias=False)
#             eye.weight.data.copy_(torch.eye(new_embeds.embedding_dim).data)
#             new_embeds = torch.nn.Sequential(new_embeds, eye)
#             model_.set_input_embeddings(new_embeds)
#             model = model_

#         elif params[8] == "gpt2-roberta": # part below unindented
###############################################################################################################
# def learn_vecmap(X, y):
#     print("computing vecmap")
#     w = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(y)
#     vecmap = torch.nn.Linear(w.size(0), w.size(1), bias=False)
#     print(w.size(), vecmap.weight.size())
#     vecmap.weight.data.copy_(w.data.t())
#     return vecmap

# def vocab_permutation(vocab1, vocab2):
#     vocab2itos = {k:v for v,k in vocab2.items()}
#     vocab2list = [vocab2itos[k] for k in range(len(vocab2itos))]

#     perm1 = []
#     perm2 = []
#     unincluded = []
#     for i, word in enumerate(vocab2list):
#         if word in vocab1:
#             perm1.append(vocab1[word])
#             perm2.append(i)
#         else:
#             unincluded.append(word)

#     print(unincluded)
#     return perm1, perm2

# embeds = model.get_input_embeddings()
# new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
# for p in new_embeds.parameters():
#     p.requires_grad = False

# new_embeds.weight.data.copy_(embeds.weight)
# print(model.device)
# config.new_n_embd = new_embeds.embedding_dim
# config.new_vocab_size = new_embeds.num_embeddings

# model_ = AutoModelForSequenceClassification.from_pretrained(params[6], config=config)
# tokenizer_ = AutoTokenizer.from_pretrained(params[6], config=config)

# perm, perm_ = vocab_permutation(tokenizer.vocab, tokenizer_.vocab)
# old_embeds = model_.get_input_embeddings()
# vecmap = learn_vecmap(new_embeds.weight[perm], old_embeds.weight[perm_])
# new_embeds = torch.nn.Sequential(new_embeds, vecmap)
# model_.set_input_embeddings(new_embeds)
# model = model_
###############################################################################################################


# else:
#     model =  AutoModelForSequenceClassification.from_pretrained(params[6], config=config)
#     model.resize_token_embeddings(len(tokenizer))
# # model =  AutoModelForSequenceClassification.from_pretrained(params[6], config=config)

###############################################################################################################
# mod_path = '/home/hyeryungson/mucoco/models_bak/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds'
# # state_dict
# mod = torch.load(os.path.join(mod_path, 'results/checkpoint-76000/pytorch_model.bin'))

# model.load_state_dict(mod)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# print(model.device)
###############################################################################################################

os.makedirs(params[7], exist_ok=True)

## optimizer 
# need to separate parameters into 2 groups to apply different weight decay values (AdamW)
# weight decay only applied to parameter weights (not to bias, layernorm weight/bias)
# code source: https://discuss.huggingface.co/t/adamw-pytorch-vs-huggingface/30562
# why?: weight decay = for overfitting https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212

# update: 23/04/21 (due to version change from transformers 4.7.1 -> 4.27.4)
# now when defining param_groups, only trainable parameters are to be included.

no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay)) and (p.requires_grad)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay)) and (p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=1e-05,
            eps=1e-08,
            weight_decay=0
)

# optim state_dict
opt = torch.load(os.path.join(ckpt_dir, "optimizer.pt"))
optimizer.load_state_dict(opt)


# load learning rate schedule
sch = torch.load(os.path.join(ckpt_dir, "scheduler.pt"))
num_training_steps = (len(train_dataset)*past_run_config['num_train_epochs']['value'])/(past_run_config['per_device_train_batch_size']['value']*(4*num_gpu))
num_warmup_steps = past_run_config['warmup_steps']['value']
last_epoch = sch['last_epoch']
print('last_epoch: ', last_epoch)
print('num_training_steps: ', num_training_steps)
lr_sch = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, last_epoch=last_epoch)
lr_sch.load_state_dict(sch)

# for i in opt['state']:
#     print(i, opt['state'][i]['exp_avg'].size())

# for i, (name, t) in enumerate(model.named_parameters()):
#     if i == 0: 
#         continue
#     if ('bias' in name) or ('LayerNorm' in name):
#         continue
#     print(i, name, ':::' , t.size())

training_args = TrainingArguments(
    output_dir=f'{params[7]}/results',          # output directory
    num_train_epochs=10,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
#     warmup_steps=600, # commented out for resume
#     weight_decay=0.01,               # strength of weight decay # commented out for resume
#     learning_rate=1e-5, # commented out for resume
    logging_dir=f'{params[7]}/logs',            # directory for storing logs
    logging_steps=100,
    evaluation_strategy="steps",
    save_total_limit=1,
    eval_steps=500,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    report_to="wandb"
)

print(training_args.n_gpu)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_sch)
)

train_result = trainer.train(resume_from_checkpoint=True)

print("training finished")

os.makedirs(f"{params[7]}/checkpoint_best")

trainer.save_model(output_dir=f"{params[7]}/checkpoint_best") 
print("model saved")

print("running evaluation now")

metrics = trainer.evaluate(val_dataset)
print("validation", metrics)
metrics = trainer.evaluate(test_dataset)
print("test", metrics)