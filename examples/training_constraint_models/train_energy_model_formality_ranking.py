# -*- coding: utf-8 -*-
import os
import sys
import math
import argparse
import logging
import json
from operator import itemgetter
sys.path.append("/home/s3/hyeryung/mucoco")
# os.chdir("/home/s3/hyeryung/mucoco")
            
from numpy import std
from tqdm import tqdm
import torch
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import torch.nn as nn

from notebooks.utils.load_ckpt import define_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                   datefmt="%m/%d/%Y %H:%M:%S", 
                   level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main(args):
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs   
    weight_decay = args.weight_decay
    filtering = args.filtering
    max_lr = args.max_lr
    model_name = args.model
    margin = args.margin
    val_loss_type = args.val_loss_type
    
    config = {'batch_size': batch_size, 
              'num_epochs': num_epochs, 
              'max_lr': max_lr, 
              'model': model_name,
              'filtering': filtering,
              'weight_decay': weight_decay,
              'margin': margin,
              'val_loss_type': val_loss_type,
              }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run = wandb.init(project='formality', entity='hayleyson', config=config)

    ### data load
    # data= pd.read_csv('data/formality/PT16/answers', delimiter='\t', names=['score', 'individual scores', 'na', 'text'])
    # data = data.sample(frac=1,random_state=999).reset_index(drop=True)#shuffle
    # train_size = math.ceil(len(data) * 0.9)

    # train_data = data.iloc[:train_size,:].copy()
    # valid_data = data.iloc[train_size:, :].copy()

    # if filtering: #only filter training data
    #     train_data['std'] = train_data['individual scores'].apply(lambda x: std([float(i) for i in str(x).split(',')]))
    #     train_data = train_data.loc[train_data['std'] < 1.5].copy()
    #     del train_data['std']

    # del train_data['individual scores']
    # del train_data['na']
    # del valid_data['individual scores']
    # del valid_data['na']

    # ## save train/valid data for reproducibility
    # if filtering:
    #     train_data.to_csv('data/formality/PT16/train_filtered.tsv', sep='\t', index=False)
    # else:
    #     train_data.to_csv('data/formality/PT16/train.tsv', sep='\t', index=False)
    # valid_data.to_csv('data/formality/PT16/valid.tsv', sep='\t', index=False)

    dpath = 'data/formality/PT16'
    if filtering:
        train_data = pd.read_csv(f'{dpath}/train_filtered.tsv', sep='\t')
    else:
        train_data = pd.read_csv(f'{dpath}/train.tsv', sep='\t')
    valid_data = pd.read_csv(f'{dpath}/valid.tsv', sep='\t')

    max_save_num = args.max_save_num
    ckpt_save_path = args.checkpoint_path #"models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered"
    os.makedirs(ckpt_save_path, exist_ok=True)
    with open(f'{ckpt_save_path}/config.txt', 'w') as f:
        config.update({'run_path': f'{run.entity}/{run.project}/{run.id}'})
        f.write(json.dumps(config))

    model, tokenizer = define_model(num_classes=1, device=device)

    class MyCollator(object):
        def __init__(self, margin):
            self.margin = margin

        def __call__(self, batch):
            ## sort samples by labels from batch
            sorted_batch = sorted(batch, key=lambda x: x['labels'])

            less_xx_batch = sorted_batch[:len(batch)//2]
            more_xx_batch = sorted_batch[len(batch)//2:]

            ## check if label difference is greater than margin (1?)
            less_xx_batch_final = []
            more_xx_batch_final = []
            for i in range(len(batch)//2):
                ## only use pairs that have label difference greater than margin
                if less_xx_batch[i]['labels'] +self.margin < more_xx_batch[i]['labels']:
                    less_xx_batch_final.append(less_xx_batch[i])
                    more_xx_batch_final.append(more_xx_batch[i])
                
            # logger.debug(f"dropped items: {len(batch)-len(less_xx_batch_final)-len(more_xx_batch_final)}")

            ## tokenize
            less_xx_outputs =  tokenizer([example['text'] for example in less_xx_batch_final], padding=True, truncation=True, return_tensors="pt")
            more_xx_outputs =  tokenizer([example['text'] for example in more_xx_batch_final], padding=True, truncation=True, return_tensors="pt")
            
            # outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
            # outputs['labels'] = torch.Tensor([example['labels'] for example in batch])
            
            return {
                "less_xx_input_ids": less_xx_outputs['input_ids'],
                "less_xx_attention_mask": less_xx_outputs['attention_mask'],
                "more_xx_input_ids": more_xx_outputs['input_ids'],
                "more_xx_attention_mask": more_xx_outputs['attention_mask'],
                "less_xx_labels": torch.Tensor([example['labels'] for example in less_xx_batch_final]),
                "more_xx_labels": torch.Tensor([example['labels'] for example in more_xx_batch_final]),
            }

    collate_fn = MyCollator(margin)
    
    def scale_labels(example):
        example["labels"] = (example["labels"] + 3.) / 6. # range: -3 ~ 3 -> 0 ~ 1
        return example
    
    train_dataset = Dataset.from_pandas(train_data)
    train_dataset = train_dataset.rename_column('score', 'labels')
    train_dataset = train_dataset.map(scale_labels)
    train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size,collate_fn=collate_fn,drop_last=True)

    valid_dataset = Dataset.from_pandas(valid_data)
    valid_dataset = valid_dataset.rename_column('score', 'labels')
    valid_dataset = valid_dataset.map(scale_labels)

    if val_loss_type == 'margin_ranking_loss':
        valid_loader = DataLoader(valid_dataset, shuffle=False,batch_size=batch_size*2,collate_fn=collate_fn,drop_last=True) 
        #c.f. batch_size*2 to avoid 1 sample in the last batch
    elif val_loss_type == 'mse_loss':
        def collate_fn_for_inference(batch):
            outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
            outputs['labels'] = torch.Tensor([example['labels'] for example in batch])
            return outputs
        valid_loader = DataLoader(valid_dataset, shuffle=False,batch_size=batch_size,collate_fn=collate_fn_for_inference)

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    num_warmup_steps = len(train_loader)*num_epochs*0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_loader)*num_epochs)

    loss_fct = nn.MarginRankingLoss(margin=margin)
    # c.f. loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)

    step=0
    best_val_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):

        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, total=len(train_loader)):

            less_xx_logits = model(input_ids = batch['less_xx_input_ids'].to(device),
                                labels = batch['less_xx_labels'].to(device),
                                attention_mask = batch['less_xx_attention_mask'].to(device)).logits

            more_xx_logits = model(input_ids = batch['more_xx_input_ids'].to(device),
                                labels = batch['more_xx_labels'].to(device),
                                attention_mask = batch['more_xx_attention_mask'].to(device)).logits

            loss = loss_fct( more_xx_logits, less_xx_logits, torch.ones_like(more_xx_logits, dtype=torch.long))
            
            loss.backward()
            train_loss += loss.item()
            
            wandb.log({'step':step, 'train_loss': loss.item(), 'learning_rate': scheduler.get_last_lr()[0]})

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step+=1


        valid_loss = 0
        for batch in valid_loader:
            model.eval()
            with torch.no_grad():
                if val_loss_type == 'margin_ranking_loss':
                    less_xx_logits = model(input_ids = batch['less_xx_input_ids'].to(device),
                                    labels = batch['less_xx_labels'].to(device),
                                    attention_mask = batch['less_xx_attention_mask'].to(device)).logits

                    more_xx_logits = model(input_ids = batch['more_xx_input_ids'].to(device),
                                        labels = batch['more_xx_labels'].to(device),
                                        attention_mask = batch['more_xx_attention_mask'].to(device)).logits

                    loss = loss_fct( more_xx_logits, less_xx_logits, torch.ones_like(more_xx_logits, dtype=torch.long))
                
                elif val_loss_type == 'mse_loss':
                    outputs = model(input_ids = batch['input_ids'].to(device),
                                    labels = batch['labels'].to(device),
                                    attention_mask = batch['attention_mask'].to(device))
                    loss = outputs.loss
                
                valid_loss += loss.item()
                
        wandb.log({'epoch':epoch, 'train_loss_per_epoch':train_loss/len(train_loader) , 'valid_loss_per_epoch': valid_loss/len(valid_loader),
                'epoch_end_learning_rate': scheduler.get_last_lr()[0]})
        logger.info(f"Epoch: {epoch}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {valid_loss/len(valid_loader)}")

        avg_val_loss = valid_loss/len(valid_loader)
        if avg_val_loss < best_val_loss:
            logger.info("Saving checkpoint!")
            best_val_loss = avg_val_loss
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': best_val_loss,
            #     },
            #     f"{ckpt_save_path}/epoch_{epoch}_best_ckpt.pt"
            # )
            tokenizer.save_pretrained(f"{ckpt_save_path}/epoch_{epoch}")
            model.save_pretrained(f"{ckpt_save_path}/epoch_{epoch}")

            # check if num of saved checkpoints exceed max_save_num.
            fileData = {}
            test_output_dir = ckpt_save_path
            for fname in os.listdir(test_output_dir):
                if fname.startswith('epoch'):
                    fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                else:
                    pass
            sortedFiles = sorted(fileData.items(), key=itemgetter(1))

            if len(sortedFiles) < max_save_num:
                pass
            else: # if so, delete the oldest checkpoint(s).
                delete = len(sortedFiles) - max_save_num
                for x in range(0, delete):
                    one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                    logger.debug(one_folder_name)
                    os.system('rm -r ' + one_folder_name)
            logger.info('-----------------------------------')
    

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Training a ranking model')
    args.add_argument('--model', type=str, default='roberta-large-ranker', help='model name')
    args.add_argument('--batch_size', type=int, default=16, help='batch size')
    args.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    args.add_argument('--max_lr', type=float, default=5e-5, help='maximum learning rate')
    args.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    args.add_argument('--filtering', action='store_true', help='filtering training data')
    args.add_argument('--margin', type=float, default=0.16666666666666666, help='margin for MarginRankingLoss & constructing batches')
    args.add_argument('--checkpoint_path', type=str, help='path to save checkpoints')
    args.add_argument('--max_save_num', type=int, default=1, help='maximum number of checkpoints to save')
    args.add_argument('--val_loss_type', type=str, default='margin_ranking_loss', choices=['margin_ranking_loss', 'mse_loss'], help='type of validation loss')
    args = args.parse_args()
    
    main(args)