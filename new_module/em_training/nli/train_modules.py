import os
import yaml
import math
import time

import json
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score, mean_squared_error
from scipy.stats import pearsonr
import seaborn as sns

from new_module.em_training.nli.losses import create_pairs_for_ranking

def load_config(config_path):
    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def plot_hist_real_num(array_positive, array_negative, path_to_save_figure):
    """
    code adapted from Mooho's code
    """
    plt.hist([array_positive, array_negative], label = [ "class 1", "class 0"])
    plt.legend(loc="upper left")
    plt.title("hist")
    plt.savefig(path_to_save_figure)
    plt.close()

def plot_roc_curve_real_num(array_positive, array_negative, path_to_save_figure):
    """
    code adapted from Mooho's code
    """
    # Combine the positive and negative arrays
    y_true = np.array([1] * len(array_positive) + [0] * len(array_negative))
    pos = [-1 * a for a in array_positive]
    neg = [-1 * a for a in array_negative]
    scores = np.array(pos + neg)
    
    # Calculate FPR and TPR for various thresholds
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auroc = roc_auc_score(y_true, scores)
    
    # Plotting the ROC curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve with AUROC={auroc:.4f}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save the plot to the specified path
    plt.savefig(path_to_save_figure)
    plt.close()
    return auroc

def plot_observed_predicted_boxplot(array_energy, array_labels, path_to_save_figure):
    predicted_result = pd.DataFrame({'labels': array_labels, 'predicted_proba': array_energy})
    predicted_result['labels_c'] = pd.cut(predicted_result['labels'], np.arange(-0.1, 1.01, 0.1))
    g = sns.boxplot(predicted_result, x='labels_c', y='predicted_proba')
    plt.xticks(rotation=30)
    plt.xticks(rotation=30)
    plt.title("Predicted Values by Bins of Observed Values")
    plt.xlabel("Bins in Labels")
    plt.ylabel("Energy")
    plt.savefig(path_to_save_figure)
    plt.close()

def validate_model(dev_dataloader, model, criterion, ranking_criterion, config, epoch, overall_step):
    model.eval()
    with torch.no_grad():
        dev_loss = 0
        dev_e = []
        dev_labels = []
        dev_fine_labels = []
        num_skipped_batch = 0
        for dev_batch in dev_dataloader:
            
            dev_predictions, hidden_states = model(input_ids = dev_batch['input_ids'],
                                               attention_mask = dev_batch['attention_mask'])
            
            if config['energynet']['loss'] == 'mse':
                dev_predictions = torch.sigmoid(dev_predictions)
                dev_loss += criterion(dev_predictions, dev_batch['labels'])
                dev_e.extend(dev_predictions.cpu().squeeze(-1).tolist())
            elif config['energynet']['loss'] in ['margin_ranking', 'negative_log_odds']:
                higher_batch, lower_batch = create_pairs_for_ranking(dev_predictions, dev_batch['labels'])
                dev_loss += criterion(dev_predictions, dev_batch['labels'])
                dev_e.extend(dev_predictions.cpu().squeeze(-1).tolist())
            elif config['energynet']['loss'] == 'mse+margin_ranking':
                # mse calculated with probability
                # margin_ranking calculated with unnormalized logit
                higher_batch, lower_batch = create_pairs_for_ranking(dev_predictions, dev_batch['labels'])
                if higher_batch.sum() == 0.0:
                    num_skipped_batch += 1
                    print("Skipping this dev set batch because only one label is included (thus could not evaluate ranking accuracy.)")
                    continue
                dev_predictions = torch.sigmoid(dev_predictions)
                dev_loss += criterion(dev_predictions, dev_batch['labels'], higher_batch, lower_batch)
                dev_e.extend(dev_predictions.cpu().squeeze(-1).tolist())
            elif config['energynet']['loss'] in ['cross_entropy', 'binary_cross_entropy']:
                dev_loss += criterion(dev_predictions, dev_batch['labels'])
                dev_predictions = torch.softmax(dev_predictions, dim=-1)
                dev_e.extend(dev_predictions.cpu()[:, config['energynet']['energy_col']].tolist()) 
            else:
                raise NotImplementedError(f"Invalid loss name {config['energynet']['loss']} provided.")
                
            if config['energynet']['loss'] in ['mse', 'margin_ranking', 'negative_log_odds', 'mse+margin_ranking']:
                dev_labels.extend(dev_batch['labels'].cpu().squeeze(-1).tolist())
            elif (config['energynet']['loss'] == 'cross_entropy') and (config['energynet']['label_column'] == 'real_num'):
                dev_labels.extend(dev_batch['labels'].cpu()[:,config['energynet']['energy_col']].tolist())
            elif (config['energynet']['loss'] == 'binary_cross_entropy') or ((config['energynet']['loss'] == 'cross_entropy') and (config['energynet']['label_column'] == 'original_labels')):
                dev_labels.extend(dev_batch['labels'].cpu().tolist())
            else:
                raise NotImplementedError("Invalid loss name provided.")
            
            if config['energynet']['add_ranking_loss']:
                if ((config['energynet']['loss'] == 'cross_entropy') and (config['energynet']['label_column'] == 'original_labels')) or \
                  ((config['energynet']['loss'] == 'binary_cross_entropy') and (config['energynet']['label_column'] == 'binary_labels')):
            
                    energy = dev_predictions[:, config['energynet']['energy_col']]
                    fine_labels = torch.Tensor(dev_batch['finegrained_labels']).to(config['device'])
                    
                    # setting 1 
                    if config['energynet']['add_ranking_loss_setting'] == 1:
                        energy = energy[~torch.isnan(fine_labels)]
                        fine_labels = fine_labels[~torch.isnan(fine_labels)]
                        
                        if len(energy) != 0:
                            
                            higher_batch, lower_batch = create_pairs_for_ranking(energy, fine_labels)
                            print(f"higher_batch size: {len(higher_batch)}")
                            print(f"lower_batch size: {len(lower_batch)}")
                            dev_loss += ranking_criterion(higher_batch, lower_batch)
                    
                    # setting 2
                    elif config['energynet']['add_ranking_loss_setting'] == 2:
                        
                        # if no finegrained labels, use crude labels
                        contradict_label = 2 if config['energynet']['label_column'] == 'original_labels' else 1
                        fine_labels[torch.isnan(fine_labels)] = torch.where(dev_batch['labels'] == contradict_label, 1., 0.)[torch.isnan(fine_labels)]
                        
                        higher_batch, lower_batch = create_pairs_for_ranking(energy, fine_labels)
                        print(f"higher_batch size: {len(higher_batch)}")
                        print(f"lower_batch size: {len(lower_batch)}")
                        dev_loss += ranking_criterion(higher_batch, lower_batch)
            
            dev_fine_labels.extend(dev_batch['finegrained_labels'])
                
        dev_loss /= (len(dev_dataloader) - num_skipped_batch)
        
        # class_0 : inconsistent (contradict) / class_1 : consistent (neutral, entail)
        if config['energynet']['label_column'] == 'finegrained_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l >= 0.5]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l < 0.5]
        elif config['energynet']['label_column'] == 'binary_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l == 1]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l == 0]
        elif config['energynet']['label_column'] == 'original_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l == 2]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l != 2]
        else:
            raise NotImplementedError
        
        # ~calculate mse~ : won't use mse because our energies aren't necessarily probabilities(0~1), while finegrained labels are probabilities.
        mse = mean_squared_error(dev_fine_labels, dev_e)
        
        # calculate pearson r
        # dev_fine_labels : close to 0 if consistent, 1 if inconsistent
        # our energy : -log (1-p(inconsistent)) ==> small if inconsistent, large if consistent
        corr_val = pearsonr(dev_fine_labels, dev_e).statistic
        
        # calculate precision, recall, f1 at threshold 0.5
        # we assume pos_label = 0 
        # dev_pred = [1 if val < 0.5 else 0 for val in dev_e]
        # dev_labels = [1 if lab == 0 else 1 for lab in dev_labels] 
        # precision, recall, f1, support = precision_recall_fscore_support(dev_labels, dev_pred, average='binary')        
        
        # precision, recall, f1 at threshold at which f1 is the maximum
        thresholds = determine_best_split_by_f1(e_class_1, e_class_0)
        threshold, precision, recall, f1 = thresholds['best_f1_threshold'], thresholds['best_f1_precision'], thresholds['best_f1_recall'], thresholds['best_f1_f1'] 
        
        # boxplot of energy against bins of finegrained labels
        plot_observed_predicted_boxplot(dev_e, dev_fine_labels, os.path.splitext(config['model_path'])[0] + '_boxplot.png')
                
        # hist plot of energy by binary labels
        plot_hist_real_num(e_class_1, e_class_0, os.path.splitext(config['model_path'])[0] + '_hist.png')

        # ROC plot
        auroc = plot_roc_curve_real_num(e_class_1, e_class_0, os.path.splitext(config['model_path'])[0] + '_ROC.png')
        
        dev_metrics = {
        'epoch': epoch, 
        'step':overall_step,
        'eval_mse': mse,
        'eval_pearsonr': corr_val,
        'eval_auroc': auroc,
        'eval_threshold': threshold,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_loss': dev_loss.item()}

    return dev_metrics

def train_model_one_step(batch, model, optimizer, scheduler, criterion, ranking_criterion, epoch, overall_step, config):
    
    model.train()
    predictions, hidden_states = model(input_ids = batch['input_ids'],
                                    attention_mask = batch['attention_mask'])
    
    if config['energynet']['loss'] == 'mse':
        predictions = torch.sigmoid(predictions)
        loss = criterion(predictions, batch['labels'])
        
    elif (config['energynet']['loss'] == "margin_ranking") or \
        (config['energynet']['loss'] == "negative_log_odds"):
        higher_batch, lower_batch = create_pairs_for_ranking(predictions, batch['labels'])
        loss = criterion(higher_batch, lower_batch)
    
    elif config['energynet']['loss'] == 'mse+margin_ranking':
        
        higher_batch, lower_batch = create_pairs_for_ranking(predictions, batch['labels'])
        predictions = torch.sigmoid(predictions)
        loss = criterion(predictions, batch['labels'], higher_batch, lower_batch)
        
    else:
        
        loss = criterion(predictions, batch['labels'])
        
        
    if config['energynet']['add_ranking_loss']:
        if ((config['energynet']['loss'] == 'cross_entropy') and (config['energynet']['label_column'] == 'original_labels')) or \
            ((config['energynet']['loss'] == 'binary_cross_entropy') and (config['energynet']['label_column'] == 'binary_labels')):
            predictions = torch.softmax(predictions, dim=-1)
            energy = predictions[:, config['energynet']['energy_col']]
            fine_labels = torch.Tensor(batch['finegrained_labels']).to(config['device'])
            
            # setting 1 
            if config['energynet']['add_ranking_loss_setting'] == 1:
                energy = energy[~torch.isnan(fine_labels)]
                fine_labels = fine_labels[~torch.isnan(fine_labels)]
                
                if len(energy) != 0:
                    
                    higher_batch, lower_batch = create_pairs_for_ranking(energy, fine_labels)
                    print(f"higher_batch size: {len(higher_batch)}")
                    print(f"lower_batch size: {len(lower_batch)}")
                    loss += ranking_criterion(higher_batch, lower_batch)
            
            # setting 2
            elif config['energynet']['add_ranking_loss_setting'] == 2:
                # if no finegrained labels, use crude labels
                contradict_label = 2 if config['energynet']['label_column'] == 'original_labels' else 1
                fine_labels[torch.isnan(fine_labels)] = torch.where(batch['labels'] == contradict_label, 1., 0.)[torch.isnan(fine_labels)]
                
                higher_batch, lower_batch = create_pairs_for_ranking(energy, fine_labels)
                print(f"higher_batch size: {len(higher_batch)}")
                print(f"lower_batch size: {len(lower_batch)}")
                loss += ranking_criterion(higher_batch, lower_batch)
            
    
    loss.backward()
    
    train_metrics = {'epoch': epoch, 
                     'step':overall_step, 
                     'train_loss': loss.item(), 
                     'learning_rate': scheduler.get_last_lr()[0]}
    
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return train_metrics

def determine_best_split_by_f1(array_positive, array_negative):
    """
    code adapted from Mooho's code
    """
    y_true = np.array([1] * len(array_positive) + [0] * len(array_negative))
    pos = [a for a in array_positive]
    neg = [a for a in array_negative]

    if (len(pos) == 0) and (len(neg) == 0):
        return {}
    if all([p==0 for p in pos]) and all([n==0 for n in neg]):
        return {}

    scores = np.array(pos + neg)

    best_precision_threshold_list, best_recall_threshold_list, best_f1_threshold_list, best_acc_threshold_list = [], [], [], []
    best_precision, best_recall, best_f1, best_accuracy = 0, 0, 0, 0
    threshold_min = float(min(scores))
    threshold_max = float(max(scores))
    difference = threshold_max - threshold_min

    threshold = threshold_min
    while threshold < threshold_max:
        threshold += difference /100
        y_pred = np.where(scores > threshold, 0, 1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # estimating f1
        if best_f1 < f1:
            best_f1 = f1
            best_f1_threshold_list = [threshold]
        elif best_f1 == f1:
            best_f1_threshold_list.append(threshold)
    
    best_f1_threshold = sum(best_f1_threshold_list) / len(best_f1_threshold_list)
    y_pred = np.where(scores > threshold, 1, 0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'best_f1_threshold': best_f1_threshold,
        'best_f1_f1': f1,
        'best_f1_precision': precision,
        'best_f1_recall': recall,
        'best_f1_accuracy': accuracy
        }
