#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import argparse
sys.path.append("/home/s3/hyeryung/mucoco")
os.chdir("/home/s3/hyeryung/mucoco")

import numpy as np
import pandas as pd
from numpy import std
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from evaluate import load
from torch.utils.data import DataLoader
from datasets import Dataset
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
import seaborn as sns

import mucoco.utils as utils
    
    
# def plot_boxplot(input_pred_data, input_pred_data_name, input_column_name, input_plot_axis, output_dir,
#                  cutoffs:np.array=np.arange(-3, 3.5, 0.5)):
    
#     import seaborn as sns
#     import matplotlib.pyplot as plt
    
#     copy_pred_data = input_pred_data[['labels', input_column_name]].copy()
#     # bin labels columns values
#     copy_pred_data['labels_cat']=pd.cut(copy_pred_data['labels'], cutoffs, include_lowest=True, right=True)
#     # copy_pred_data['mod_proba_cat']=pd.cut(copy_pred_data[input_column_name], cutoffs, include_lowest=True, right=True)
    
#     sns.boxplot(data = copy_pred_data, x = 'labels_cat', y=input_column_name, ax=input_plot_axis)
#     input_plot_axis.set_title(f'Test data: {input_pred_data_name}, Model: {input_column_name}')

#     input_plot_axis.set_ylabel('')
#     plt.savefig(os.path.join(output_dir, f'boxplot_{input_pred_data_name}_{input_column_name}.png'), 
#                 dpi=300, bbox_inches='tight')

def predict_labels(args, device):
    config = AutoConfig.from_pretrained(args.checkpoint_dir)
    model = utils.RobertaCustomForSequenceClassification.from_pretrained(args.checkpoint_dir, config=config)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    def scale_labels(value):
        value = (value + 3.) / 6. # range: -3 ~ 3 -> 0 ~ 1
        return value

    test_data = pd.read_csv(args.test_data_path, sep="\t")
    if args.rescale_labels:
        test_data["score"] = test_data["score"].apply(scale_labels)

    test_dataset = Dataset.from_pandas(test_data)
    test_dataset = test_dataset.rename_column('score', 'labels')

    def collate_fn(batch):
        outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
        outputs['labels'] = torch.Tensor([example['labels'] for example in batch])
        return outputs
    
    test_loader = DataLoader(test_dataset, shuffle=False,batch_size=args.batch_size,collate_fn=collate_fn)

    predictions = []
    for batch in test_loader:
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'].to(device), 
                            labels = batch['labels'].to(device),
                            attention_mask = batch['attention_mask'].to(device))
            predictions.extend(outputs.logits.reshape(-1,).tolist())


    labels_predictions = pd.DataFrame({"predictions": predictions, "labels": test_data['score'].tolist()})
    labels_predictions.to_csv(os.path.join(args.output_dir, "labels_predictions.csv"))
    return labels_predictions

def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(args.output_dir, "labels_predictions.csv")):
        print("labels_predictions.csv already exists. Skipping...")
        labels_predictions = pd.read_csv(os.path.join(args.output_dir, "labels_predictions.csv"))
    else:
        print("Predicting labels...")
        labels_predictions = predict_labels(args, device)
        print("Done.")


    ### Plot & Analyze Model Outputs

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"RMSE: {mean_squared_error(labels_predictions['labels'], labels_predictions['predictions'])**(1/2)}\n")
        f.write(f"MAE: {mean_absolute_error(labels_predictions['labels'], labels_predictions['predictions'])}\n")
        f.write(f"Pearson's r: {pearsonr(labels_predictions['labels'], labels_predictions['predictions'])[0]}\n")
        f.write(f"Min prediction: {min(labels_predictions['predictions'])}\n")
        f.write(f"Max prediction: {max(labels_predictions['predictions'])}\n")

    # labels_predictions['labels_cat']=pd.cut(labels_predictions['labels'], np.arange(-3, 3.5, 0.5), include_lowest=False, right=False)
    # rmse_by_bin=[]
    # mae_by_bin=[]

    # rmse = labels_predictions.groupby('labels_cat').apply(lambda x: round(mean_squared_error(x['labels'], x['predictions'])**(1/2), 2))
    # rmse.name = f'rmse'
    # mae = labels_predictions.groupby('labels_cat').apply(lambda x: round(mean_absolute_error(x['labels'], x['predictions']),2))
    # mae.name = f'mae'
    # rmse_by_bin.append(rmse)
    # mae_by_bin.append(mae)


    # ----------------------------------- #
    # Create a confusion matrix
    if args.rescale_labels:
        # if labels_predictions['predictions'].max() > 1. or labels_predictions['predictions'].min() < 0.:
        #     print('Warning: predictions are not in the range of 0 ~ 1')
        #     pmin = labels_predictions['predictions'].min()
        #     pmax = labels_predictions['predictions'].max()
        #     interval = (pmax - pmin) / 4 
        #     labels_predictions['pred_cat']=pd.cut(labels_predictions['predictions'], np.arange(pmin, pmax+0.1, interval), include_lowest=True, right=True)
        # else:    
        labels_predictions['pred_cat']=pd.cut(labels_predictions['predictions'], np.arange(0, 1.1, 0.25), include_lowest=True, right=True)
        
        labels_predictions['labels_cat']=pd.cut(labels_predictions['labels'], np.arange(0, 1.1, 0.25), include_lowest=True, right=True)        
    else:
        labels_predictions['pred_cat']=pd.cut(labels_predictions['predictions'], np.arange(-3, 3.1, 1.5), include_lowest=True, right=True)
        labels_predictions['labels_cat']=pd.cut(labels_predictions['labels'], np.arange(-3, 3.1, 1.5), include_lowest=True, right=True)


    print('count groupby labels_cat')
    print(labels_predictions.groupby(['labels_cat']).size())
    print('count groupby pred_cat')
    print(labels_predictions.groupby(['pred_cat']).size())

    cm = labels_predictions.groupby(['pred_cat', 'labels_cat']).size().unstack(0).sort_index(ascending=False)
    print('confusion matrix')
    print(cm)
    
    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues')#, fmt='d')

    # Add labels, title, and axis ticks
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.clf()


    # ----------------------------------- #
    # Create a confusion matrix with recall
    cm = labels_predictions.groupby(['pred_cat', 'labels_cat']).size().unstack(0).apply(lambda x: x/x.sum(),axis=1).sort_index(ascending=False)
    print('confusion matrix')
    print(cm)


    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues')#, fmt='d')

    # Add labels, title, and axis ticks
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Recall)')

    # Show the plot
    plt.show()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix_recall.png"), dpi=300, bbox_inches='tight')
    plt.clf()
    
    # ----------------------------------- #
    # Plot predictions by bin
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
    # if args.rescale_labels:
    #     plot_boxplot(labels_predictions, args.test_data_type, 'predictions', axes, args.output_dir, cutoffs=np.arange(0, 1.1, 0.25))
    # else:
    #     plot_boxplot(labels_predictions, args.test_data_type, 'predictions', axes, args.output_dir, cutoffs=np.arange(-3, 3.1,1.5))
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
    sns.boxplot(data = labels_predictions, x = 'pred_cat', y='labels', ax=axes)
    axes.set_title(f'Test data: {args.test_data_type}, Model: labels')
    axes.set_ylabel('')
    plt.savefig(os.path.join(args.output_dir, f'boxplot_{args.test_data_type}_labels.png'), 
                dpi=300, bbox_inches='tight')
    plt.clf()
    
    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
    sns.boxplot(data = labels_predictions, x = 'labels_cat', y='predictions', ax=axes2)
    axes2.set_title(f'Test data: {args.test_data_type}, Model: predictions')
    axes2.set_ylabel('')
    plt.savefig(os.path.join(args.output_dir, f'boxplot_{args.test_data_type}_predictions.png'), 
                dpi=300, bbox_inches='tight')
    plt.clf()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--test_data_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default='regression')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rescale_labels", action='store_true')
    args = parser.parse_args()

    main(args)

