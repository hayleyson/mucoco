import os
import yaml
import math
import time
import shutil

import json
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW, SGD, RMSprop, Adam, NAdam

import numpy as np
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
import seaborn as sns

from new_module.em_training.nli.models import EncoderModel
from new_module.em_training.nli.data_handling import load_nli_data, load_additional_nli_training_data, NLI_Dataset, NLI_DataLoader, NLI_TrainBatchSampler_Binary
from new_module.em_training.nli.train_modules import *
from new_module.em_training.nli.losses import create_pairs_for_ranking, CustomMarginRankingLoss, PairwiseLogisticLoss, MSE_MarginRankingLoss

def main():
    
    config = load_config('new_module/em_training/config_xy_concat.yaml')
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## add more elaborate dirs in ckpt_save_path
    model_dir_1 = f"{config['energynet']['base_model']}_{os.path.splitext(config['energynet']['dataset_path'])[0].split('/')[-1]}_{config['energynet']['label_column']}_{config['energynet']['loss']}".replace('-', '_')
    model_dir_2 = str(int(time.time()))
    config['energynet']['ckpt_save_path'] = f"{config['energynet']['ckpt_save_path']}/{model_dir_1}/{model_dir_2}"
    model_path = f"{config['energynet']['ckpt_save_path']}/best_model.pth"
    config['model_path'] = model_path
    
    runname = f"{config['energynet']['base_model']}_{config['energynet']['output_form']}_{config['energynet']['label_column']}_{config['energynet']['loss']}_{model_dir_2}"
    run = wandb.init(config=config, entity="hayleyson", project="nli_energynet")
    run_config = wandb.config
    try:
        config['energynet']['batch_size'] = run_config.batch_size
        config['energynet']['max_lr'] = run_config.max_lr
        config['energynet']['weight_decay'] = run_config.weight_decay
        config['energynet']['num_epochs'] = run_config.num_epochs
    except:
        pass
    wandb.run.name = runname
    
    model = EncoderModel(config)
    model = model.to(config['device'])
    
    max_lr = float(config['energynet']['max_lr'])
    weight_decay = float(config['energynet']['weight_decay'])
    num_epochs = int(config['energynet']['num_epochs'])
    
    if not os.path.exists(config['energynet']['ckpt_save_path']):
        os.makedirs(config['energynet']['ckpt_save_path'])
    
    train_dev_data = load_nli_data(output_file_path=config['energynet']['dataset_path'])
    
    if config['energynet']['add_train_data']:
        train_add_data = load_additional_nli_training_data(output_file_path='data/nli/snli_mnli_anli_train_without_finegrained.jsonl')
        if (config['energynet'].get('fill_missing_finegrained') is not None) and (config['energynet']['fill_missing_finegrained']):
            print(f"Num missing finegrained labels before filling: {train_add_data['finegrained_labels'].isna().sum()}")
            train_add_data.loc[train_add_data['finegrained_labels'].isna(), 'finegrained_labels'] = train_add_data.loc[train_add_data['finegrained_labels'].isna(), 'binary_labels']
            print(f"Num missing finegrained labels after filling: {train_add_data['finegrained_labels'].isna().sum()}")
        train_dev_data = pd.concat([train_dev_data, train_add_data], axis=0)

    train_data = train_dev_data.loc[train_dev_data['split'] == 'train']
    dev_data = train_dev_data.loc[train_dev_data['split'] == 'dev']
    dev_data = dev_data.sample(frac=1, random_state=0) # shuffle rows to make sure margin ranking loss works.
    
    print(f"# train samples: {len(train_data)}, # dev samples: {len(dev_data)}")
    
    ## IMPT. 'finegrained_labels' == degree of contradiction (real number between 0 and 1) == portion of annotators who labeled the sample as "contradiction"
    train_dataset = NLI_Dataset(train_data, label_column=config['energynet']['label_column'])
    dev_dataset = NLI_Dataset(dev_data, label_column=config['energynet']['label_column'])
    
    if config['energynet']['binary_loader'] == 'balanced':
        binary_batchsampler = NLI_TrainBatchSampler_Binary(train_data, config['energynet']['batch_size']['binary'], oversample_minority = True)
        train_dataloader = NLI_DataLoader(config = config,
                                            tokenizer = model.tokenizer).get_dataloader(train_dataset, batch_size=None, batch_sampler=binary_batchsampler)
    else:
        train_dataloader = NLI_DataLoader(config = config,
                                        tokenizer = model.tokenizer).get_dataloader(train_dataset, batch_size=config['energynet']['batch_size']['binary'], batch_sampler=None, shuffle=True)
    
    
    dev_dataloader = NLI_DataLoader(config = config,
                                     tokenizer = model.tokenizer).get_dataloader(dev_dataset, batch_size=config['energynet']['batch_size']['continuous'], batch_sampler=None, shuffle=False)
    
    # Define the optimizer
    try:
        if run_config.optimizer=='sgd':
            optimizer = SGD(model.parameters(),lr=max_lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        elif run_config.optimizer=='rmsprop':
            optimizer = RMSprop(model.parameters(),lr=max_lr, weight_decay=weight_decay)
        elif run_config.optimizer=='adam':
            optimizer = Adam(model.parameters(),lr=max_lr, betas=(0.9, 0.999))
        elif run_config.optimizer=='nadam':
            optimizer = NAdam(model.parameters(),lr=max_lr,  betas=(0.9, 0.999))
        elif run_config.optimizer=='adamw':
            optimizer = AdamW(model.parameters(),lr=max_lr, weight_decay=weight_decay)
    except:
        optimizer = AdamW(model.parameters(),lr=max_lr, weight_decay=weight_decay)

    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps*0.1, num_training_steps=num_training_steps)
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_dataloader)*num_epochs,
                                                                #    num_cycles=num_epochs)
    
    
    if (config['energynet']['loss'] == 'cross_entropy') or (config['energynet']['loss'] == 'binary_cross_entropy'):
        criterion = nn.CrossEntropyLoss()
    elif config['energynet']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['energynet']['loss'] == 'margin_ranking':
        criterion = CustomMarginRankingLoss(margin=config['energynet']['margin'])
    elif config['energynet']['loss'] in ['pairwise_logistic', 'scaled_ranking', 'negative_log_odds']:
        criterion = PairwiseLogisticLoss()
    elif config['energynet']['loss'] == 'mse+margin_ranking':
        criterion = MSE_MarginRankingLoss(weights = [1.,1.], 
                                          margin=config['energynet']['margin'])
    else:
        raise NotImplementedError('Not a valid loss name')
        
    overall_step = 0
    # eval_metric = 'pearsonr'
    # eval_goal = 'maximize'
    # best_val_metric = float('inf') if eval_goal == 'minimize' else -1000.
    eval_metrics = ['ndcg', 'pearsonr','pearsonr_subset', 'loss']
    eval_goals = ['maximize', 'maximize','maximize', 'minimize']
    best_val_metrics = [float('inf') if eval_goal == 'minimize' else -1000. for eval_goal in eval_goals]
    
    # Early stopping parameters
    patience = config['energynet']['early_stopping']['patience'] # Number of eval_every iterations to wait before stopping
    min_delta = config['energynet']['early_stopping']['min_delta'] # Minimum change in loss to qualify as an improvement
    patience_counter = 0
    best_loss = None
    
    # log_file = open(f"{config['energynet']['ckpt_save_path']}/log.txt", 'w')
    for epoch in tqdm(range(num_epochs)):
        
        for i, batch in enumerate(train_dataloader):
            
            train_metrics = train_model_one_step(batch, model, optimizer, scheduler, criterion, epoch, overall_step, config)
            overall_step += 1
            
            if overall_step % config['energynet']['eval_every'] == 0:
                
                dev_metrics = validate_model(dev_dataloader, model, criterion, config, epoch, overall_step)
                all_metrics = {**dev_metrics, **train_metrics}
                wandb.log(all_metrics)
                try: 
                    wandb.log({"boxplot": wandb.Image(os.path.dirname(config['model_path']) + '/current_model_boxplot.png', caption=f"{runname}")}, step = overall_step)
                    wandb.log({"hist": wandb.Image(os.path.dirname(config['model_path']) + '/current_model_hist.png', caption=f"{runname}")}, step = overall_step)
                    wandb.log({"ROC": wandb.Image(os.path.dirname(config['model_path']) + '/current_model_ROC.png', caption=f"{runname}")}, step = overall_step)
                except:
                    useless_variable = 1
                    
                for idx in range(len(eval_metrics)):
                    if ((eval_goals[idx] == 'maximize') and (dev_metrics[f"eval_{eval_metrics[idx]}"] > best_val_metrics[idx])) or \
                        ((eval_goals[idx] == 'minimize') and (dev_metrics[f"eval_{eval_metrics[idx]}"] < best_val_metrics[idx])) :
                        
                        best_val_metrics[idx] = dev_metrics[f"eval_{eval_metrics[idx]}"]
                        torch.save(model.state_dict(), model_path.split('.pth')[0] + f"_{eval_metrics[idx]}.pth")
                        
                        f = open(model_path.split('.pth')[0] + f"_{eval_metrics[idx]}_info.txt", 'w')
                        json.dump(dev_metrics, f)
                        f.close()
                        
                        shutil.copy2(os.path.dirname(config['model_path']) + '/current_model_boxplot.png',
                                     model_path.split('.pth')[0] + f"_{eval_metrics[idx]}_boxplot.png")
                        
                        shutil.copy2(os.path.dirname(config['model_path']) + '/current_model_hist.png',
                                     model_path.split('.pth')[0] + f"_{eval_metrics[idx]}_hist.png")
                        
                        shutil.copy2(os.path.dirname(config['model_path']) + '/current_model_ROC.png',
                                     model_path.split('.pth')[0] + f"_{eval_metrics[idx]}_ROC.png")
                    
                # Check for improvement
                if config["energynet"]["early_stopping"]["use"]:
                    
                    if best_loss is None:
                        best_loss = dev_metrics["eval_loss"]
                        print(f"Updating best loss at iteration {i}")
                    elif dev_metrics["eval_loss"] >= best_loss - min_delta:
                        if dev_metrics['eval_loss'] < best_loss:
                            best_loss = dev_metrics['eval_loss']
                            print(f"Updating best loss at iteration {i}")
                        patience_counter += 1
                        if (patience_counter >= patience):
                            print(f"Early stopping at epoch {epoch}")
                            break
                    else:
                        best_loss = dev_metrics["eval_loss"]
                        patience_counter = 0
            else:
                wandb.log(train_metrics)
    
    

# ToDo. 무호코드 참고해서 sweep 하는 부분 추가하기
if __name__ == "__main__":
    
    
    # Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
    sweep_config = {
        'method': 'random', #grid, random
        'early_terminate':{
            'type': 'hyperband',
            's': 2, # max number of unique executions of Successive Halving
            'eta': 3, # downsampling rate for Successive Halving
            'max_iter': 27 # max iterations per configuration
        },
        'metric': {
        'name': 'eval_accuracy',
        'goal': 'maximize'   
        },
        'parameters': {
            'num_epochs': {
                'values': [10, 15]
            },
            'batch_size': {
                'values': [16]
            },
            'weight_decay': {
                'values': [0.01]
            },
            'max_lr': {
                'values': [3e-5, 1e-5]
            },
            'optimizer': {
                'values': ['adamw', 'adam']
            },
        }
    }
    
    # sweep_id = wandb.sweep(sweep_config, entity="hayleyson", project="nli_energynet")
    # sw_count = math.prod([len(val['values']) for val in sweep_config['parameters'].values()])
    # wandb.agent(sweep_id, function=main, count=sw_count//2)
    # wandb.agent(sweep_id, function=main)
    main()
    
    
    
    