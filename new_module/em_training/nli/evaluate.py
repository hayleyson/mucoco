import argparse
import os
os.chdir('/data/hyeryung/mucoco')
import yaml
import datetime

import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
import seaborn as sns
from sklearn.metrics import ndcg_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score

from new_module.em_training.nli.models import EncoderModel
from new_module.em_training.nli.data_handling import load_nli_data, load_nli_test_data, NLI_Dataset, NLI_DataLoader
from new_module.em_training.nli.train import *
from new_module.em_training.nli.train_modules import *


def main(args):
    
    api = wandb.Api()
    run = api.run(args.wandb_run_path)
    config = run.config

    assert ((args.ckpt_criterion !='') and (not os.path.exists(config['model_path']))) or (os.path.exists(config['model_path']))
    
    if args.ckpt_criterion == '':
        model_path = config['model_path']
    else:
        model_path = config['model_path'].split('.pth')[0] + f'_{args.ckpt_criterion}.pth'
    
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = EncoderModel(config)
    model = model.to(config['device'])
    model.load_state_dict(torch.load(model_path))

    # load meta
    try:
        with open(model_path.replace('.pth', '_info.txt'), 'r') as f:
            meta_info = json.load(f)
    except:
        meta_info = {}
    
    # define data loader 
    train_dev_data = load_nli_data(output_file_path=config['energynet']['dataset_path'])
    dev_data = train_dev_data.loc[train_dev_data['split'] == 'dev']
    dev_dataset = NLI_Dataset(dev_data, label_column=config['energynet']['label_column'])
    dev_dataloader = NLI_DataLoader(config = config, 
                                    tokenizer=model.tokenizer).get_dataloader(dev_dataset, batch_size=config['energynet']['batch_size'], batch_sampler=None, shuffle=False)
    
    assert len(dev_dataloader) == 351
    
    # define loss function b/c validate_model requires it
    if (config['energynet']['loss'] == 'cross_entropy') or (config['energynet']['loss'] == 'binary_cross_entropy'):
        criterion = nn.CrossEntropyLoss()
    elif config['energynet']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['energynet']['loss'] == 'margin_ranking':
        criterion = CustomMarginRankingLoss(margin=config['energynet']['margin'])
    elif config['energynet']['loss'] == 'negative_log_odds':
        criterion = PairwiseLogisticLoss()
    elif config['energynet']['loss'] == 'mse+margin_ranking':
        criterion = MSE_MarginRankingLoss(weights = [1.,1.], 
                                          margin=config['energynet']['margin'])
    else:
        raise NotImplementedError('Not a valid loss name')
    
    
    # inference
    config['legacy'] = args.legacy
    eval_metrics = validate_model(dev_dataloader, model, criterion, config, -1, -1)
    eval_metrics['run_id'] = args.wandb_run_path.split('/')[-1]
    eval_metrics['step'] = meta_info.get('step',None)
    eval_metrics['epoch'] = meta_info.get('epoch',None)
    
    return eval_metrics
    
    

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--wandb_run_path',type=str)
    # parser.add_argument('--ckpt_criterion',type=str,default='')
    # parser.add_argument('--legacy',action='store_true')
    # args = parser.parse_args()
    
    class Args:
        def __init__(self, arg1, arg2, arg3=False):
            
            self.wandb_run_path = arg1
            self.ckpt_criterion = arg2
            self.legacy = arg3
    
    # run_ids = ["t6lbryef"]
    run_ids="""ikwakty4
csadvpu5
2iak1bw5
wps90kq3""".split()

    # ckpt_criterions = ["add_ranking_loss" if rid in ["8ct1dplo", "twj4g8f9"] else "" for rid in run_ids]
    ckpt_criterions = ["continuous_loss"] * len(run_ids)

    results_df = []
    for run_id, criterion in zip(run_ids, ckpt_criterions):
        
        args = Args(f"hayleyson/nli_energynet/{run_id}", criterion)
        results_df.append(main(args))
        
    results_df = pd.DataFrame(results_df)
    results_df.to_excel(f"new_module/em_training/nli/evaluation_results/nli_energynet_metrics_{datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')}_2.xlsx", index=False)