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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score, mean_squared_error, ndcg_score
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
    predicted_result['labels_c'] = pd.cut(predicted_result['labels'], np.arange(predicted_result['labels'].min()-0.1, predicted_result['labels'].max()+0.01, 0.1))
    g = sns.boxplot(predicted_result, x='labels_c', y='predicted_proba')
    plt.xticks(rotation=30)
    plt.xticks(rotation=30)
    plt.title("Predicted Values by Bins of Observed Values")
    plt.xlabel("Bins in Labels")
    plt.ylabel("Predicted Labels or Energy")
    plt.savefig(path_to_save_figure)
    plt.close()

def plot_observed_predicted_boxplot_v2(array_energy, array_labels, path_to_save_figure):
    predicted_result = pd.DataFrame({'labels': array_labels, 'predicted_proba': array_energy})
    cutoffs = np.linspace(predicted_result.labels.min()-0.1, 
                      predicted_result.loc[predicted_result.labels < 20, 'labels'].max(), 
                      num=10)
    cutoffs = np.append(cutoffs, np.array([predicted_result['labels'].max()]))
    cutoffs = np.round(cutoffs, 2)
    cutoffs = pd.IntervalIndex.from_breaks(cutoffs, closed='right')
    predicted_result['labels_c'] = pd.cut(predicted_result['labels'], cutoffs)
    g = sns.boxplot(predicted_result, x='labels_c', y='predicted_proba')
    plt.xticks(rotation=30)
    plt.xticks(rotation=30)
    plt.title("Predicted Values by Bins of Observed Values")
    plt.xlabel("Bins in Labels")
    plt.ylabel("Energy")
    plt.savefig(path_to_save_figure)
    plt.close()

def validate_model_loss_mix(dev_dataloader, model, binary_criterion, continuous_criterion, mixing_weights, config, epoch, overall_step):
    
    model.eval()
    with torch.no_grad():
        dev_loss = 0
        dev_binary_loss = 0
        dev_continuous_loss = 0
        dev_e = []
        dev_labels = []
        dev_fine_labels = []
        num_skipped_batch = 0
        
        for dev_batch in dev_dataloader:
            
            dev_predictions, _ = model(input_ids = dev_batch['input_ids'],
                                       attention_mask = dev_batch['attention_mask'])
            dev_e_ = -torch.log_softmax(dev_predictions, dim=-1)[:, config['energynet']['energy_col']]
            dev_e.extend(dev_e_.tolist()) 
            dev_labels.extend(dev_batch['labels'].cpu().tolist())
            dev_fine_labels.extend(dev_batch['finegrained_labels'].tolist())
            
            binary_loss = binary_criterion(dev_predictions, dev_batch['labels'])
            dev_binary_loss += binary_loss
            
            if config['energynet']['additional_loss']['loss'] in ["margin_ranking", "pairwise_logistic"]:
                higher_batch, lower_batch = create_pairs_for_ranking(-dev_e_, dev_batch['finegrained_labels'])
                
                if higher_batch.sum() == 0.0:
                    num_skipped_batch += 1
                    continue
                continuous_loss = continuous_criterion(higher_batch, lower_batch)
                
            elif config['energynet']['additional_loss']['loss'] == "cross_entropy":
                labels = dev_batch['finegrained_labels'].reshape(-1, 1)
                labels = torch.tile(labels, (1,2))
                labels[:, 0] = 1 - labels[:, 1]
                continuous_loss = continuous_criterion(dev_predictions, labels)
            
            dev_loss += (mixing_weights[0] * binary_loss + mixing_weights[1] * continuous_loss)
            dev_continuous_loss += continuous_loss
                
        dev_loss /= (len(dev_dataloader) - num_skipped_batch)
        dev_binary_loss /= (len(dev_dataloader))
        dev_continuous_loss /= (len(dev_dataloader) - num_skipped_batch)
        
        # class_0 : inconsistent (contradict) / class_1 : consistent (neutral, entail)
        if config['energynet']['label_column'] == 'binary_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l == 0]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l == 1]
        elif config['energynet']['label_column'] == 'original_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l == 2]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l != 2]
        else:
            raise NotImplementedError
        
        dev_fine_labels_for_metrics = torch.Tensor(dev_fine_labels)
        dev_fine_labels_for_metrics[dev_fine_labels_for_metrics == 0] = 1e-10
        dev_fine_labels_for_metrics = (-torch.log(dev_fine_labels_for_metrics)).tolist()
        
        mse = mean_squared_error(dev_fine_labels_for_metrics, dev_e)
        corr_val = pearsonr(dev_fine_labels_for_metrics, dev_e).statistic
        
        # calculate corr only using examples with labels between 0 and 1 exclusive
        dev_e_subset = [e for e, l in zip(dev_e, dev_fine_labels) if (l < 1) and (0 < l)]
        dev_fine_labels_subset = (-torch.log(torch.Tensor([l for l in dev_fine_labels if (l < 1) and (0 < l)]))).tolist()
        corr_subset_val = pearsonr(dev_fine_labels_subset, dev_e_subset).statistic
        
        ndcg_val = ndcg_score([dev_fine_labels_for_metrics], [dev_e])
        ndcg_subset_val = ndcg_score([dev_fine_labels_subset], [dev_e_subset])
        
        # calculate precision, recall, f1 at threshold 0.5
        # we assume pos_label = 0 
        # dev_pred = [1 if val < 0.5 else 0 for val in dev_e]
        # dev_labels = [1 if lab == 0 else 1 for lab in dev_labels] 
        # precision, recall, f1, support = precision_recall_fscore_support(dev_labels, dev_pred, average='binary')        
        
        # precision, recall, f1 at threshold at which f1 is the maximum
        thresholds = determine_best_split_by_f1(e_class_1, e_class_0)
        threshold, precision, recall, f1 = thresholds['best_f1_threshold'], thresholds['best_f1_precision'], thresholds['best_f1_recall'], thresholds['best_f1_f1'] 
        
        # boxplot of energy against bins of finegrained labels
        plot_observed_predicted_boxplot_v2(dev_e, dev_fine_labels_for_metrics, os.path.dirname(config['model_path']) + '/current_model_boxplot.png')
                
        # hist plot of energy by binary labels
        plot_hist_real_num(e_class_1, e_class_0, os.path.dirname(config['model_path']) + '/current_model_hist.png')

        # ROC plot
        auroc = plot_roc_curve_real_num(e_class_1, e_class_0, os.path.dirname(config['model_path']) + '/current_model_ROC.png')
        
        dev_metrics = {
        'epoch': epoch, 
        'step':overall_step,
        'eval_mse': mse,
        'eval_pearsonr': corr_val,
        'eval_pearsonr_subset': corr_subset_val,
        'eval_ndcg': ndcg_val,
        'eval_ndcg_subset': ndcg_subset_val,
        'eval_auroc': auroc,
        'eval_threshold': threshold,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_loss': dev_loss.item(),
        'eval_binary_loss': dev_binary_loss.item(),
        'eval_continuous_loss': dev_continuous_loss.item()}
        
    return dev_metrics

def validate_model(dev_dataloader, model, criterion, config, epoch, overall_step):
    model.eval()
    with torch.no_grad():
        dev_loss = 0
        dev_e = []
        dev_labels = []
        dev_fine_labels = []
        num_skipped_batch = 0
        
        for dev_batch in dev_dataloader:
            
            dev_predictions, _ = model(input_ids = dev_batch['input_ids'],
                                        attention_mask = dev_batch['attention_mask'])
            
            if config['energynet']['loss'] in ['mse', 'mse+margin_ranking']:
                dev_predictions = torch.sigmoid(dev_predictions)
                dev_loss += criterion(dev_predictions, dev_batch['labels'])
                dev_e.extend((-torch.log(dev_predictions)).cpu().squeeze(-1).tolist())
            elif config['energynet']['loss'] in ['margin_ranking', 'pairwise_logistic', 'negative_log_odds']:
                higher_batch, lower_batch = create_pairs_for_ranking(dev_predictions, dev_batch['labels'])
                if higher_batch.sum().item() == 0:
                    num_skipped_batch += 1
                    continue
                dev_loss += criterion(higher_batch, lower_batch)
                dev_e.extend((-dev_predictions).cpu().squeeze(-1).tolist())
            elif config['energynet']['loss'] in ['cross_entropy', 'binary_cross_entropy']:
                dev_loss += criterion(dev_predictions, dev_batch['labels'])
                if (config.get('legacy', False)) and (config['energynet']['output_form'] == '2dim_vec'):
                    # legacy models with 2dim_vec output used to be trained as index 1 representing contradiction proba. dev_e should be consistency proba.
                    dev_e.extend((-torch.log_softmax(dev_predictions, dim=-1)).cpu()[:, 1 - config['energynet']['energy_col']].tolist()) 
                elif config['energynet']['output_form'] == '3dim_vec':
                    # defining energy_col for 3dim_vec is a bit tricky. since entail and neutral are both consistent, energy_col should point to contradict class(2), so that we can use 1-p(contradict)
                    # legacy models with 2dim_vec output used to be trained as index 1 representing contradiction proba. dev_e should be consistency proba.
                    dev_e.extend((-torch.log(1-torch.softmax(dev_predictions, dim=-1)[:, config['energynet']['energy_col']])).cpu().tolist()) 
                else:
                    dev_e.extend((-torch.log_softmax(dev_predictions, dim=-1)).cpu()[:, config['energynet']['energy_col']].tolist()) 
            else:
                raise NotImplementedError(f"Invalid loss name {config['energynet']['loss']} provided.")
                
            if config['energynet']['loss'] in ['margin_ranking', 'pairwise_logistic', 'negative_log_odds']:
                dev_labels.extend(dev_batch['labels'].cpu().squeeze(-1).tolist())
            elif config['energynet']['loss'] in ['mse', 'mse+margin_ranking']:
                dev_labels.extend(dev_batch['labels'].cpu().squeeze(-1).tolist())
            elif (config['energynet']['output_form'] == '2dim_vec') and (config['energynet']['label_column'] == 'finegrained_labels'):
                dev_labels.extend(dev_batch['labels'].cpu()[:,config['energynet']['energy_col']].tolist())
            elif (config['energynet']['label_column'] == 'binary_labels') or (config['energynet']['label_column'] == 'original_labels'):
                dev_labels.extend(dev_batch['labels'].cpu().tolist())
            else:
                raise NotImplementedError("Invalid loss name provided.")
            
            dev_fine_labels.extend(dev_batch['finegrained_labels'].tolist())
                
        dev_loss /= (len(dev_dataloader) - num_skipped_batch)
        
        # class_0 : inconsistent (contradict) / class_1 : consistent (neutral, entail)
        if config['energynet']['label_column'] == 'finegrained_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l < 0.5]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l >= 0.5]
        elif config['energynet']['label_column'] == 'binary_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l == 0]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l == 1]
        elif config['energynet']['label_column'] == 'original_labels':
            e_class_0 = [e for e, l in zip(dev_e, dev_labels) if l == 2]
            e_class_1 = [e for e, l in zip(dev_e, dev_labels) if l != 2]
        else:
            raise NotImplementedError
        
        dev_fine_labels_for_metrics = torch.Tensor(dev_fine_labels)        
        if config['energynet']['loss'] in ['cross_entropy', 'binary_cross_entropy', 'mse', 'mse+margin_ranking']:
            dev_fine_labels_for_metrics[dev_fine_labels_for_metrics == 0] = 1e-10
            dev_fine_labels_for_metrics = (-torch.log(dev_fine_labels_for_metrics)).tolist()
        elif config['energynet']['loss'] in ['margin_ranking', 'pairwise_logistic', 'negative_log_odds', 'scaled_ranking']:
            dev_fine_labels_for_metrics = (-dev_fine_labels_for_metrics).tolist()
        else:
            raise NotImplementedError
            
        mse = mean_squared_error(dev_fine_labels_for_metrics, dev_e)
        corr_val = pearsonr(dev_fine_labels_for_metrics, dev_e).statistic
        if config['energynet']['loss'] in ['margin_ranking', 'pairwise_logistic', 'negative_log_odds', 'scaled_ranking']:
            # ndcg_score function cannot take negative y_true
            ndcg_val = ndcg_score([[-1*x for x in dev_fine_labels_for_metrics]], [[-1*x for x in dev_e]])
        else:
            ndcg_val = ndcg_score([dev_fine_labels_for_metrics], [dev_e])
        
        # calculate metrics only using examples with labels between 0 and 1 exclusive
        dev_e_subset = [e for e, l in zip(dev_e, dev_fine_labels) if (l < 1) and (0 < l)]
        dev_fine_labels_subset = [e for e, l in zip(dev_fine_labels_for_metrics, dev_fine_labels) if (l < 1) and (0 < l)]
        
        corr_subset_val = pearsonr(dev_fine_labels_subset, dev_e_subset).statistic
        if config['energynet']['loss'] in ['margin_ranking', 'pairwise_logistic', 'negative_log_odds', 'scaled_ranking']:
            # ndcg_score function cannot take negative y_true
            ndcg_subset_val = ndcg_score([[-1*x for x in dev_fine_labels_subset]], [[-1*x for x in dev_e_subset]])
        else:
            ndcg_subset_val = ndcg_score([dev_fine_labels_subset], [dev_e_subset])
        
        # calculate precision, recall, f1 at threshold 0.5
        # we assume pos_label = 0 
        # dev_pred = [1 if val < 0.5 else 0 for val in dev_e]
        # dev_labels = [1 if lab == 0 else 1 for lab in dev_labels] 
        # precision, recall, f1, support = precision_recall_fscore_support(dev_labels, dev_pred, average='binary')        
        
        # precision, recall, f1 at threshold at which f1 is the maximum
        # thresholds = determine_best_split_by_f1(e_class_1, e_class_0)
        # threshold, precision, recall, f1 = thresholds['best_f1_threshold'], thresholds['best_f1_precision'], thresholds['best_f1_recall'], thresholds['best_f1_f1'] 
        
        # boxplot of energy against bins of finegrained labels
        if config['energynet']['loss'] in ['margin_ranking', 'pairwise_logistic', 'negative_log_odds']:
            plot_observed_predicted_boxplot(dev_e, dev_fine_labels_for_metrics, os.path.dirname(config['model_path']) + '/current_model_boxplot.png')
        else:
            plot_observed_predicted_boxplot_v2(dev_e, dev_fine_labels_for_metrics, os.path.dirname(config['model_path']) + '/current_model_boxplot.png')
                
        # hist plot of energy by binary labels
        plot_hist_real_num(e_class_1, e_class_0, os.path.dirname(config['model_path']) + '/current_model_hist.png')

        # ROC plot
        auroc = plot_roc_curve_real_num(e_class_1, e_class_0, os.path.dirname(config['model_path']) + '/current_model_ROC.png')
        
        dev_metrics = {
        'epoch': epoch, 
        'step':overall_step,
        'eval_mse': mse,
        'eval_pearsonr': corr_val,
        'eval_pearsonr_subset': corr_subset_val,
        'eval_ndcg': ndcg_val,
        'eval_ndcg_subset': ndcg_subset_val,
        'eval_auroc': auroc,
        # 'eval_threshold': threshold,
        # 'eval_precision': precision,
        # 'eval_recall': recall,
        # 'eval_f1': f1,
        'eval_loss': dev_loss.item()}

    return dev_metrics

def train_model_one_step_loss_mix(binary_batch, continuous_batch, model, optimizer, scheduler, binary_criterion, continuous_criterion, mixing_weights, epoch, overall_step, config):
    
    model.train()
    # inference
    # binary_predictions, _ = model(input_ids = binary_batch['input_ids'],
    #                                 attention_mask = binary_batch['attention_mask'])
    continuous_predictions, _ = model(input_ids = continuous_batch['input_ids'],
                                attention_mask = continuous_batch['attention_mask'])
    
    binary_loss = binary_criterion(continuous_predictions, continuous_batch['labels'])
    # if config['energynet']['additional_loss']['loss'] == 'cross_entropy':
    #     binary_loss = binary_criterion(binary_predictions, binary_batch['labels'])
    # else:
    #     # binary loss
    #     binary_loss = binary_criterion(torch.cat((binary_predictions, continuous_predictions), dim=0), 
    #                                 torch.cat((binary_batch['labels'], continuous_batch['labels']), dim=0))
        
    energy = -torch.log_softmax(continuous_predictions, dim=-1)[:, config['energynet']['energy_col']]
    labels = continuous_batch['finegrained_labels']
    higher_batch, lower_batch = create_pairs_for_ranking(-energy, labels)
    continuous_loss = continuous_criterion(higher_batch, lower_batch)
    # # continuous loss
    # if config['energynet']['additional_loss']['setting'] == 2: ## assume binary labels indicate uninanimous votes and treat them as continuous labels
    #     if config['energynet']['additional_loss']['loss'] == 'margin_ranking':
        
    #         predictions = torch.cat((binary_predictions, continuous_predictions), dim=0)
    #         energy = -torch.log_softmax(predictions, dim=-1)[:, config['energynet']['energy_col']]
    #         labels = torch.cat((binary_batch['labels'].float(), continuous_batch['finegrained_labels']),dim=0)
    #         higher_batch, lower_batch = create_pairs_for_ranking(-energy, labels)
    #         continuous_loss = continuous_criterion(higher_batch, lower_batch)
            
    #     elif config['energynet']['additional_loss']['loss'] == 'pairwise_logistic':
            
    #         predictions = torch.cat((binary_predictions, continuous_predictions), dim=0)
    #         # print(f"shape of predictions: {predictions.shape}")
    #         # predictions = predictions[:, 1] - predictions[:, 0] # using difference in logits as signals
    #         # print(f"shape of predictions after taking difference in logits: {predictions.shape}")
    #         energy = -torch.log_softmax(predictions, dim=-1)[:, config['energynet']['energy_col']]
    #         labels = torch.cat((binary_batch['labels'].float(), continuous_batch['finegrained_labels']),dim=0)
    #         higher_batch, lower_batch = create_pairs_for_ranking(-energy, labels)
    #         continuous_loss = continuous_criterion(higher_batch, lower_batch)
        
    #     else:
    #         raise NotImplementedError
        
    # elif config['energynet']['additional_loss']['setting'] == 3:
    #     if config['energynet']['additional_loss']['loss'] == 'cross_entropy':
    #         labels = continuous_batch['finegrained_labels'].reshape(-1, 1)
    #         labels = torch.tile(labels, (1,2))
    #         labels[:, 0] = 1 - labels[:, 1]
    #         continuous_loss = continuous_criterion(continuous_predictions, labels)
    #     elif config['energynet']['additional_loss']['loss'] == 'margin_ranking':    
    #         energy = -torch.log_softmax(continuous_predictions, dim=-1)[:, config['energynet']['energy_col']]
    #         higher_batch, lower_batch = create_pairs_for_ranking(-energy, continuous_batch['finegrained_labels'])
    #         continuous_loss = continuous_criterion(higher_batch, lower_batch)
    #     elif config['energynet']['additional_loss']['loss'] == 'pairwise_logistic':
    #         # print(f"shape of continuous predictions: {continuous_predictions.shape}")
    #         continuous_predictions = continuous_predictions[:, 1] - continuous_predictions[:, 0]
    #         # print(f"shape of continuous predictions after taking difference in logits: {continuous_predictions.shape}")
    #         higher_batch, lower_batch = create_pairs_for_ranking(continuous_predictions, continuous_batch['finegrained_labels'])
    #         continuous_loss = continuous_criterion(higher_batch, lower_batch)
    #     else:
    #         raise NotImplementedError
    
    # add losses together
    loss = mixing_weights[0] * binary_loss + mixing_weights[1] * continuous_loss    
    loss.backward()
    
    train_metrics = {'epoch': epoch, 
                     'step': overall_step, 
                     'train_loss': loss.item(), 
                     'train_binary_loss': binary_loss.item(),
                     'train_continuous_loss': continuous_loss.item(),
                     'learning_rate': scheduler.get_last_lr()[0]}
    
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return train_metrics

def train_model_one_step(batch, model, optimizer, scheduler, criterion, epoch, overall_step, config):
    
    model.train()
    if config['energynet']['input_form'] == 'xy_concat':
        half_count = len(batch['input_ids'])//2
        ## OOM error can arise because xy_concat data collator produces 2x samples given initial set of samples
        ## run inference twice
        predictions_1, hidden_states = model(input_ids = batch['input_ids'][:half_count],
                                        attention_mask = batch['attention_mask'][:half_count])
        predictions_2, hidden_states = model(input_ids = batch['input_ids'][half_count:],
                                attention_mask = batch['attention_mask'][half_count:])
        predictions = torch.cat([predictions_1, predictions_2], dim=0)

    else:   
        predictions, hidden_states = model(input_ids = batch['input_ids'],
                                        attention_mask = batch['attention_mask'])
    
    if config['energynet']['loss'] == 'mse': # energy = - log_sigmoid(predictions)
        loss = criterion(torch.sigmoid(predictions), batch['labels'])
    elif ('ranking' in config['energynet']['loss']): # energy = - predictions b/c higher f(x) -> lower energy
        energy = -predictions
        higher_batch, lower_batch = create_pairs_for_ranking(-energy, batch['labels'])
        loss = criterion(higher_batch, lower_batch)
    else: # cross_entropy, bce # energy = - log_softmax(predictions)[:, 1] b/c higher log p -> lower energy
        loss = criterion(predictions, batch['labels'])
        
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
