# -*- coding: utf-8 -*-
import os
import sys
import math
import argparse
import logging
import json
import time
from operator import itemgetter
# sys.path.append("/home/s3/hyeryung/mucoco")
# os.chdir("/home/s3/hyeryung/mucoco")
sys.path.append(".")
os.chdir(".")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            
from tqdm import tqdm
import torch
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import torch.nn as nn

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from new_module.utils.load_ckpt import define_model

logger = get_logger(__name__, log_level = os.environ.get("LOGGING_LEVEL", "ERROR"))

def create_pairs_for_ranking(logits, labels):
    """ Given model predictions(logits or probabilities) and ground truth labels of a list of examples, 
    fetch all possible pairs from the examples, compare the ground truth values of each pair, 
    and return two lists where the first list contains the model predictions for items in pairs that have higher g.t. and
    the second list that contains the predictions for items with lower g.t..
    """
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

class NegativeLogOddsLoss(nn.Module):
    """
    Implementation of loss from "Learning to Summarize from Human Feedback" paper.
    """
    
    def __init__(self):
        super(NegativeLogOddsLoss, self).__init__()

    def forward(self, fy_i, fy_1_i):
        return - torch.mean(torch.log(torch.sigmoid(fy_i - fy_1_i)))

class CustomMarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(CustomMarginRankingLoss, self).__init__()
        self.margin = margin
        self.loss_func = nn.MarginRankingLoss(margin=self.margin)
    
    def forward(self, fy_i, fy_1_i):
        return self.loss_func(fy_i, fy_1_i, torch.ones_like(fy_i, dtype=torch.long))
        

def validate_model(model, accelerator, eval_dataloader, args):
    "Compute performance of the model on the validation dataset"
    "ToDo: check if it's conventional to use loss functions for metrics."
    "ToDo: check if this implementation yields the same result as using evaluate.metrics."

    model.eval()
    
    if args.val_loss_type == 'margin_ranking_loss':
        metrics_func = CustomMarginRankingLoss(margin=args.margin)
    elif args.val_loss_type == 'scaled_ranking_loss':
        metrics_func = NegativeLogOddsLoss()
    elif args.val_loss_type == 'mse_loss':
        metrics_func = nn.MSELoss()
    
    valid_loss = 0.
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'],
                            labels = batch['labels'],
                            attention_mask = batch['attention_mask'])
        if (model.module.num_labels == 1):
            predictions = torch.sigmoid(outputs.logits)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

        elif (model.module.num_labels == 2):
            predictions = torch.softmax(outputs.logits, dim = -1)
            predictions, references = accelerator.gather_for_metrics((predictions[:, 1], batch["labels"][:, 1]))
            
        if "ranking" in args.val_loss_type:
            
            higher_batch, lower_batch = create_pairs_for_ranking(predictions, references)
            metrics = metrics_func(higher_batch, lower_batch)
        else:
            metrics = metrics_func(predictions, references)
        
        valid_loss += (metrics.item()*len(references)) # mean over batch * batch_size = sum over batch
        
    return valid_loss/len(eval_dataloader.dataset)


def main(args):
    
    start_time = time.time()
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs   
    weight_decay = args.weight_decay
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
    training_loss_type = args.training_loss_type
    model_type = args.model_type
    
    config = {'batch_size': batch_size, 
              'num_epochs': num_epochs, 
              'max_lr': max_lr, 
              'model': model_name,
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

    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device
    
    wandb_init_kwargs = {"wandb": {"entity": wandb_entity, "notes": ckpt_save_path}}
    if args.resume_from_checkpoint is not None:
        wandb_init_kwargs["wandb"].update({"id": args.resume_wandb_id, "resume": "must"})
    accelerator.init_trackers(project_name=wandb_project, config=config, init_kwargs=wandb_init_kwargs)
    run = accelerator.get_tracker("wandb")
        
    os.makedirs(ckpt_save_path, exist_ok=True)
    with open(f'{ckpt_save_path}/config.txt', 'w') as f:
        f.write(json.dumps(config))

    num_classes = 2 if training_loss_type == "cross_entropy" else 1
        
    if model_type == "RobertaCustomForSequenceClassification": 
        model, tokenizer = define_model(num_classes=num_classes, device=device)
    elif model_type == "AutoModelForSequenceClassification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    def collate_fn_default(batch):
        outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
        outputs['labels'] = torch.Tensor([example['labels'] for example in batch])
        return outputs
    
    def collate_fn_bce(batch):
        outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
        outputs['labels'] = torch.Tensor([[1-example['labels'], example['labels']] for example in batch])
        return outputs
    
    if model.num_labels == 1:
        collate_fn = collate_fn_default
    elif model.num_labels == 2:
        collate_fn = collate_fn_bce
        
    with accelerator.main_process_first():
        if train_data_path.endswith('.tsv'):
            train_data = pd.read_csv(train_data_path, sep='\t')
        elif train_data_path.endswith('.jsonl'):
            train_data = pd.read_json(train_data_path, lines=True)
            
        if valid_data_path.endswith('.tsv'):
            valid_data = pd.read_csv(valid_data_path, sep='\t')
        elif valid_data_path.endswith('.jsonl'):
            valid_data = pd.read_json(valid_data_path, lines=True)
        
        train_dataset = Dataset.from_pandas(train_data)
        valid_dataset = Dataset.from_pandas(valid_data)

    train_dataloader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size,collate_fn=collate_fn,drop_last=True)
    # update: 23/01/09: noticed that collate_fn most likely go with the train collate_fn. 
    # will take care of edge case later.
    eval_dataloader = DataLoader(valid_dataset, shuffle=False,batch_size=batch_size,collate_fn=collate_fn) 


    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    num_warmup_steps = len(train_dataloader)*num_epochs*0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_dataloader)*num_epochs)
    print(f"Initial lr: {scheduler.get_last_lr()[0]}")

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    ## some of the following code taken from https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py
    overall_step = 0
    starting_epoch = 0
    resume_step = 0
    best_val_loss = float("inf")
    best_val_step = -1

    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
            print(f"Initial lr after loading state: {scheduler.get_last_lr()[0]}")
        else:
            # Get the most recent checkpoint
            dirs = [os.path.join(ckpt_save_path, f.name) for f in os.scandir(ckpt_save_path) 
                    if f.is_dir() and f.name.startswith("laststep_")]
            dirs.sort(key=os.path.getctime)
            accelerator.print(f"Resumed from checkpoint: {dirs[-1]}")
            accelerator.load_state(dirs[-1])
            print(f"Initial lr after loading state: {scheduler.get_last_lr()[0]}")
            path = os.path.basename(dirs[-1])  # Sorts folders by date modified, most recent checkpoint is the last
            
        
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        resume_step = int(training_difference.replace("laststep_", ""))
        overall_step = resume_step ## set overall_step to resume_step
        starting_epoch = resume_step // len(train_dataloader) ## calculate starting epoch
        resume_step -= starting_epoch * len(train_dataloader) ## calculate starting step within the epoch
    ## end code excerpt
        
        best_val_loss_info = torch.load(os.path.join(ckpt_save_path, path, "best_val_info.pt"))
        best_val_loss = best_val_loss_info['best_val_loss']
        best_val_step = best_val_loss_info['best_val_step']

    
    if ("ranking" in training_loss_type) and (weight_ranking > 0.0) and args.ranking_loss_type == 'margin_ranking_loss':
        ranking_loss_fct = CustomMarginRankingLoss(margin=args.margin)
    elif ("ranking" in training_loss_type) and (weight_ranking > 0.0) and args.ranking_loss_type == 'scaled_ranking_loss':
        ranking_loss_fct = NegativeLogOddsLoss()
    else:
        ranking_loss_fct = None
        
    if num_validate_steps == -1:
        num_validate_steps = len(train_dataloader)

    ## setting seed for reproducibility in data_loaders!
    set_seed(42)
    for epoch in tqdm(range(starting_epoch, num_epochs),
                      disable=not accelerator.is_local_main_process):

        model.train()
        train_loss = 0
        
        if args.resume_from_checkpoint is not None and epoch == starting_epoch and resume_step > 0:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
            
        for i, batch in tqdm(enumerate(active_dataloader), 
                             total=len(active_dataloader),
                             disable=not accelerator.is_local_main_process):
            
            outputs = model(input_ids = batch['input_ids'],
                            labels = batch['labels'],
                            attention_mask = batch['attention_mask'])

            if training_loss_type == "cross_entropy":
                loss = outputs.loss
            elif training_loss_type == "mse":
                loss = outputs.loss
            elif training_loss_type == "ranking":
                higher_batch, lower_batch = create_pairs_for_ranking(predictions, references)
                loss = ranking_loss_fct(higher_batch, lower_batch)
            elif training_loss_type == "mse+ranking":
                mse_loss = outputs.loss
                higher_batch, lower_batch = create_pairs_for_ranking(predictions, references)
                ranking_loss = ranking_loss_fct(higher_batch, lower_batch)
                loss = weight_mse * mse_loss + weight_ranking * ranking_loss
            
            # loss.backward()
            accelerator.backward(loss)
            train_loss += loss.item()
            
            train_metrics = {'step':overall_step, 'train_loss': loss.item(), 'learning_rate': scheduler.get_last_lr()[0]}
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            overall_step+=1
            
            if overall_step % num_validate_steps == 0:
                
                valid_loss = validate_model(model, accelerator, eval_dataloader, args)
                valid_metrics = {'epoch':epoch, 'valid_loss': valid_loss}
                accelerator.log({**train_metrics, **valid_metrics}, step=overall_step)
                
                avg_val_loss = valid_loss
                if avg_val_loss < best_val_loss:
                    logger.info("Saving checkpoint!")
                    best_val_loss = avg_val_loss
                    best_val_step = overall_step
                    
                    accelerator.wait_for_everyone()
                    tokenizer.save_pretrained(f"{ckpt_save_path}/step_{overall_step}")
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(f"{ckpt_save_path}/step_{overall_step}")
                    
                    with open(f"{ckpt_save_path}/step_{overall_step}/val_loss.txt", 'w') as f:
                        f.write(str(best_val_loss))
                    
                    accelerator.save_state(f"{ckpt_save_path}/step_{overall_step}")
                    torch.save({"best_val_loss": best_val_loss,
                                "best_val_step": best_val_step
                                }, os.path.join(f"{ckpt_save_path}/step_{overall_step}", "best_val_info.pt"))
                        
                    if accelerator.is_main_process:
                        # check if num of saved checkpoints exceed max_save_num.
                        fileData = {}
                        test_output_dir = ckpt_save_path
                        for fname in os.listdir(test_output_dir):
                            # if fname.startswith('epoch'):
                            if fname.startswith('step'):
                                # fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                                fileData[fname] = open(f"{test_output_dir}/{fname}/val_loss.txt", "r").read()
                            else:
                                pass
                        # sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                        sortedFiles = sorted(fileData.items(), key=itemgetter(1), reverse=True)

                        if len(sortedFiles) < max_save_num:
                            pass
                        else: # if so, delete the checkpoint(s) with the highest val_loss
                            delete = len(sortedFiles) - max_save_num
                            for x in range(0, delete):
                                one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                                logger.debug(one_folder_name)
                                os.system('rm -r ' + one_folder_name)
                        logger.info('-----------------------------------')
            else:
                # wandb.log(train_metrics)
                accelerator.log(train_metrics, step=overall_step)
            
            accelerator.wait_for_everyone()
            # save the last step for resuming training
            output_dir = f"{ckpt_save_path}/laststep_{overall_step}"
            accelerator.save_state(output_dir)
            torch.save({"best_val_loss": best_val_loss,
                        "best_val_step": best_val_step
                        }, os.path.join(output_dir, "best_val_info.pt"))
            
            if overall_step > 1 and accelerator.is_main_process:
                last_output_dir = f"{ckpt_save_path}/laststep_{overall_step-1}"
                logger.debug(last_output_dir)
                os.system('rm -r ' + last_output_dir)

            if (time.time() - start_time) >= 60*60*args.hours_limit:
                logger.info(f"Stopping training due to time limit. Stopping at step: {overall_step}")
                accelerator.end_training()
                 

    logger.info(f"best_val_loss: {best_val_loss}, best_val_step: {best_val_step}")
    os.rename(f"{ckpt_save_path}/step_{best_val_step}", f"{ckpt_save_path}/step_{best_val_step}_best_checkpoint")
    accelerator.end_training()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to train an energy model')
    
    parser.add_argument('--task', type=str, default='formality', choices=['formality', 'sentiment', 'toxicity'], help='task name')
    parser.add_argument('--model', type=str, default='roberta-base', help='model name')
    parser.add_argument('--model_type', type=str, help='type of model', choices=['AutoModelForSequenceClassification', 'RobertaCustomForSequenceClassification'])
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--max_lr', type=float, default=5e-5, help='maximum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

    parser.add_argument('--training_loss_type', type=str, choices=['cross_entropy', 'mse', 'ranking', 'mse+ranking'])
    parser.add_argument('--loss_weight_mse', type=float, default=0., help='weight for mse loss')
    parser.add_argument('--loss_weight_ranking', type=float, default=0., help='weight for ranking loss')
    parser.add_argument('--ranking_loss_type', type=str, default='margin_ranking_loss', choices=['margin_ranking_loss', 'scaled_ranking_loss'], help='type of ranking loss')
    parser.add_argument('--margin', type=float, default=0.16666666666666666, help='margin for MarginRankingLoss & constructing batches')
    parser.add_argument('--val_loss_type', type=str, default='margin_ranking_loss', choices=['margin_ranking_loss', 'scaled_ranking_loss', 'mse_loss'], help='type of validation loss')
    
    parser.add_argument('--train_data_path', type=str, help='training data path')
    parser.add_argument('--valid_data_path', type=str, help='validation data path')
    
    parser.add_argument('--checkpoint_path', type=str, help='path to save checkpoints')
    parser.add_argument('--max_save_num', type=int, default=1, help='maximum number of checkpoints to save')
    parser.add_argument('--num_validate_steps', type=int, default=100, help='number of steps until validate & save checkpoint. -1: only validate & save checkpoint at the end of each epoch')
        
    parser.add_argument('--wandb_entity', type=str, default='hayleyson', help='wandb entity')
    parser.add_argument('--wandb_project', type=str, default='formality', help='wandb project')    
    parser.add_argument('--resume_from_checkpoint', type=str, help='directory of checkpoint to resume training from')
    parser.add_argument('--resume_wandb_id', type=str, help='wandb id to resume training from')
    parser.add_argument('--hours_limit', type=int, default=47, help='number of hours at which voluntarily stop the training considering the time limit in the servers the script is run.')
    
    args = parser.parse_args()
    if (args.training_loss_type == "cross_entropy"):
        args.ranking_loss_type = None
        args.loss_weight_mse = None
        args.loss_weight_ranking = None
        args.margin = None
        
    main(args)