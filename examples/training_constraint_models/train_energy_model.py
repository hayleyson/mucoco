# -*- coding: utf-8 -*-
import os
import sys
import math
import argparse
import logging
import json
from operator import itemgetter
sys.path.append("/home/s3/hyeryung/mucoco")
os.chdir("/home/s3/hyeryung/mucoco")
            
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

def create_pairs_for_ranking(labels, logits):

    ## what would be an efficient to create pairs?
    first = []
    second = []
    num_samples = len(labels)
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            if labels[i] > labels[j]:
                first.append(i)
                second.append(j)
            elif labels[i] < labels[j]:
                first.append(j)
                second.append(i)
    better_logits = logits[first]
    worse_logits = logits[second]

    return better_logits, worse_logits

class MyCollator(object):
    def __init__(self, margin, tokenizer):
        self.margin = margin
        self.tokenizer = tokenizer

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
        less_xx_outputs =  self.tokenizer([example['text'] for example in less_xx_batch_final], padding=True, truncation=True, return_tensors="pt")
        more_xx_outputs =  self.tokenizer([example['text'] for example in more_xx_batch_final], padding=True, truncation=True, return_tensors="pt")
        
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
        
def scale_labels_pt16(example):
    example["labels"] = (example["labels"] + 3.) / 6. # range: -3 ~ 3 -> 0 ~ 1
    return example

def scale_labels_yelp(example):
    example["labels"] = (example["labels"] - 1.) / 4. # range: 1 ~ 5 -> 0 ~ 1
    return example



class NegativeLogOddsLoss(nn.Module):
    def __init__(self):
        super(NegativeLogOddsLoss, self).__init__()

    def forward(self, fy_i, fy_1_i):
        return - torch.mean(torch.log(torch.sigmoid(fy_i - fy_1_i)))

def validate_model(model, valid_loader, val_loss_type, ranking_loss_fct, device):
    "Compute performance of the model on the validation dataset"
    valid_loss = 0.
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            if val_loss_type == 'margin_ranking_loss':
                
                less_xx_logits = model(input_ids = batch['less_xx_input_ids'].to(device),
                                labels = batch['less_xx_labels'].to(device),
                                attention_mask = batch['less_xx_attention_mask'].to(device)).logits

                more_xx_logits = model(input_ids = batch['more_xx_input_ids'].to(device),
                                    labels = batch['more_xx_labels'].to(device),
                                    attention_mask = batch['more_xx_attention_mask'].to(device)).logits
                
                loss = ranking_loss_fct( more_xx_logits, less_xx_logits, torch.ones_like(more_xx_logits, dtype=torch.long))
            
            elif val_loss_type == 'scaled_ranking_loss':
                outputs = model(input_ids = batch['input_ids'].to(device),
                                labels = batch['labels'].to(device),
                                attention_mask = batch['attention_mask'].to(device))
                more_xx_logits, less_xx_logits = create_pairs_for_ranking(batch['labels'], outputs.logits)

                loss = ranking_loss_fct(more_xx_logits, less_xx_logits)
            
            elif val_loss_type == 'mse_loss':
                outputs = model(input_ids = batch['input_ids'].to(device),
                                labels = batch['labels'].to(device),
                                attention_mask = batch['attention_mask'].to(device))
                loss = outputs.loss
            
            valid_loss += (loss.item()*len(batch['labels']))
    return valid_loss/len(valid_loader)


def main(args):
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs   
    weight_decay = args.weight_decay
    #filtering = args.filtering
    max_lr = args.max_lr
    model_name = args.model
    margin = args.margin
    val_loss_type = args.val_loss_type
    weight_mse = args.loss_weight_mse
    weight_ranking = args.loss_weight_ranking
    ranking_loss_type = args.ranking_loss_type
    max_save_num = args.max_save_num
    ckpt_save_path = args.checkpoint_path #"models/roberta-base-pt16-formality-ranker-with-gpt2-large-embeds-filtered"  
    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project  
    num_validate_steps = args.num_validate_steps
    
    config = {'batch_size': batch_size, 
              'num_epochs': num_epochs, 
              'max_lr': max_lr, 
              'model': model_name,
              #'filtering': filtering,
              'weight_decay': weight_decay,
              'margin': margin,
              'val_loss_type': val_loss_type,
              'loss_weight_mse': weight_mse,
              'loss_weight_ranking': weight_ranking,
              'ranking_loss_type': ranking_loss_type,
              'train_data_path': train_data_path,
              'valid_data_path': valid_data_path,
              }
    if ranking_loss_type == "scaled_ranking_loss":
        del config['margin']

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run = wandb.init(project=wandb_project, entity=wandb_entity, config=config, notes=ckpt_save_path)

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

    if train_data_path.endswith('.tsv'):
        train_data = pd.read_csv(train_data_path, sep='\t')
    elif train_data_path.endswith('.jsonl'):
        train_data = pd.read_json(train_data_path, lines=True)
        
    if valid_data_path.endswith('.tsv'):
        valid_data = pd.read_csv(valid_data_path, sep='\t')
    elif valid_data_path.endswith('.jsonl'):
        valid_data = pd.read_json(valid_data_path, lines=True)
        
    os.makedirs(ckpt_save_path, exist_ok=True)
    with open(f'{ckpt_save_path}/config.txt', 'w') as f:
        config.update({'run_path': f'{run.entity}/{run.project}/{run.id}',
                       'run_name': run.name})
        f.write(json.dumps(config))

    model, tokenizer = define_model(num_classes=1, device=device)

    def collate_fn_default(batch):
        outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
        outputs['labels'] = torch.Tensor([example['labels'] for example in batch])
        return outputs
    
    collate_fn = collate_fn_default
    if (ranking_loss_type == "margin_ranking_loss") and (weight_ranking != 0.0):
        collate_fn = MyCollator(margin, tokenizer)
    
    train_dataset = Dataset.from_pandas(train_data)
    #train_dataset = train_dataset.rename_column('score', 'labels')
    if args.task == 'formality':
        train_dataset = train_dataset.map(scale_labels_pt16)
    elif args.task == 'sentiment':
        train_dataset = train_dataset.map(scale_labels_yelp)
    train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size,collate_fn=collate_fn,drop_last=True)

    valid_dataset = Dataset.from_pandas(valid_data)
    #valid_dataset = valid_dataset.rename_column('score', 'labels')
    if args.task == 'formality':
        valid_dataset = valid_dataset.map(scale_labels_pt16)
    elif args.task == 'sentiment':
        valid_dataset = valid_dataset.map(scale_labels_yelp)

    if val_loss_type == 'margin_ranking_loss':
        valid_loader = DataLoader(valid_dataset, shuffle=False,batch_size=batch_size*2,collate_fn=collate_fn,drop_last=True) 
        #c.f. batch_size*2 to avoid 1 sample in the last batch
    elif val_loss_type == 'scaled_ranking_loss':
        valid_loader = DataLoader(valid_dataset, shuffle=False,batch_size=batch_size,collate_fn=collate_fn) 
    elif val_loss_type == 'mse_loss':
        valid_loader = DataLoader(valid_dataset, shuffle=False,batch_size=batch_size,collate_fn=collate_fn_default)

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    num_warmup_steps = len(train_loader)*num_epochs*0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_loader)*num_epochs)

    if ranking_loss_type == "margin_ranking_loss":
        ranking_loss_fct = nn.MarginRankingLoss(margin=1)
    elif ranking_loss_type == "scaled_ranking_loss":
        ranking_loss_fct = NegativeLogOddsLoss()
        
    mse_loss_fct = nn.MSELoss()
    # loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)
    if num_validate_steps == -1:
        num_validate_steps = n_steps_per_epoch

    step=0
    best_val_loss = float("inf")
    best_val_step = -1
    for epoch in tqdm(range(num_epochs)):

        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            if weight_ranking == 0:
                outputs = model(input_ids = batch['input_ids'].to(device),
                            labels = batch['labels'].to(device),
                            attention_mask = batch['attention_mask'].to(device))
                mse_loss = outputs.loss
                ranking_loss = -1.
            else:
                if ranking_loss_type == "margin_ranking_loss":
                    less_xx_logits = model(input_ids = batch['less_xx_input_ids'].to(device),
                                        labels = batch['less_xx_labels'].to(device),
                                        attention_mask = batch['less_xx_attention_mask'].to(device)).logits

                    more_xx_logits = model(input_ids = batch['more_xx_input_ids'].to(device),
                                        labels = batch['more_xx_labels'].to(device),
                                        attention_mask = batch['more_xx_attention_mask'].to(device)).logits

                    
                    ranking_loss = ranking_loss_fct( more_xx_logits, less_xx_logits, torch.ones_like(more_xx_logits, dtype=torch.long))

                    all_logits = torch.cat([less_xx_logits.squeeze(), more_xx_logits.squeeze()], dim=0).to(device)
                    all_labels = torch.cat([batch['less_xx_labels'].squeeze(), batch['more_xx_labels'].squeeze()], dim=0).to(device)

                    mse_loss = mse_loss_fct(all_logits, all_labels)
                    
                elif ranking_loss_type == "scaled_ranking_loss":
                    outputs = model(input_ids = batch['input_ids'].to(device),
                            labels = batch['labels'].to(device),
                            attention_mask = batch['attention_mask'].to(device))

                    mse_loss = outputs.loss

                    ## create pairs for ranking loss
                    more_xx_logits, less_xx_logits = create_pairs_for_ranking(batch['labels'], outputs.logits)

                    # ranking_loss = ranking_loss_fct( more_xx_logits, less_xx_logits, torch.ones_like(more_xx_logits, dtype=torch.long))
                    ranking_loss = ranking_loss_fct(more_xx_logits, less_xx_logits)

            logger.debug(f"mse_loss: {mse_loss}, ranking_loss: {ranking_loss}")
            logger.debug(f"weight_mse *  mse_loss: {weight_mse *  mse_loss}, weight_ranking * ranking_loss: {weight_ranking * ranking_loss}")
            loss = weight_mse *  mse_loss + weight_ranking * ranking_loss
                
            loss.backward()
            train_loss += loss.item()
            
            train_metrics = {'step':step, 'train_loss': loss.item(), 'learning_rate': scheduler.get_last_lr()[0]}
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step+=1
        
            if step % num_validate_steps == 0:
                
                valid_loss = validate_model(model, valid_loader, val_loss_type, ranking_loss_fct, device) ## also check correlation?
                valid_metrics = {'epoch':epoch, 'valid_loss': valid_loss}
                wandb.log({**train_metrics, **valid_metrics})
                
                # avg_val_loss = valid_loss/len(valid_loader)
                avg_val_loss = valid_loss
                if avg_val_loss < best_val_loss:
                    logger.info("Saving checkpoint!")
                    best_val_loss = avg_val_loss
                    best_val_step = step
                    # tokenizer.save_pretrained(f"{ckpt_save_path}/epoch_{epoch}")
                    # model.save_pretrained(f"{ckpt_save_path}/epoch_{epoch}")
                    tokenizer.save_pretrained(f"{ckpt_save_path}/step_{step}")
                    model.save_pretrained(f"{ckpt_save_path}/step_{step}")
                    torch.save({"optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),}, f"{ckpt_save_path}/step_{step}/optimizer_scheduler.pt")

                    # check if num of saved checkpoints exceed max_save_num.
                    fileData = {}
                    test_output_dir = ckpt_save_path
                    for fname in os.listdir(test_output_dir):
                        # if fname.startswith('epoch'):
                        if fname.startswith('step'):
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
            else:
                wandb.log(train_metrics)

    logger.info(f"best_val_loss: {best_val_loss}, best_val_step: {best_val_step}")
    os.rename(f"{ckpt_save_path}/step_{step}", f"{ckpt_save_path}/step_{step}_best_checkpoint")
        
    

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Training a ranking model')
    args.add_argument('--model', type=str, default='roberta-large-ranker', help='model name')
    args.add_argument('--batch_size', type=int, default=16, help='batch size')
    args.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    args.add_argument('--max_lr', type=float, default=5e-5, help='maximum learning rate')
    args.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    #args.add_argument('--filtering', action='store_true', help='filtering training data')
    args.add_argument('--margin', type=float, default=0.16666666666666666, help='margin for MarginRankingLoss & constructing batches')
    args.add_argument('--checkpoint_path', type=str, help='path to save checkpoints')
    args.add_argument('--max_save_num', type=int, default=1, help='maximum number of checkpoints to save')
    args.add_argument('--val_loss_type', type=str, default='margin_ranking_loss', choices=['margin_ranking_loss', 'scaled_ranking_loss', 'mse_loss'], help='type of validation loss')
    args.add_argument('--loss_weight_mse', type=float, default=0., help='weight for mse loss')
    args.add_argument('--loss_weight_ranking', type=float, default=1., help='weight for ranking loss')
    args.add_argument('--ranking_loss_type', type=str, default='margin_ranking_loss', choices=['margin_ranking_loss', 'scaled_ranking_loss'], help='type of ranking loss')
    args.add_argument('--train_data_path', type=str, help='training data path')
    args.add_argument('--valid_data_path', type=str, help='validation data path')
    args.add_argument('--wandb_entity', type=str, default='hayleyson', help='wandb entity')
    args.add_argument('--wandb_project', type=str, default='formality', help='wandb project')
    args.add_argument('--task', type=str, default='formality', choices=['formality', 'sentiment'], help='task name')
    args.add_argument('--num_validate_steps', type=int, default=-1, help='number of steps until validate & save checkpoint. -1: only validate & save checkpoint at the end of each epoch')
    
    args = args.parse_args()
    
    main(args)