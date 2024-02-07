#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import argparse
import re



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


def predict_labels(args, device):
    
    try:
        config = AutoConfig.from_pretrained(args.checkpoint_dir)
        if 'roberta-base-custom' == args.model_type:
            model = utils.RobertaCustomForSequenceClassification.from_pretrained(args.checkpoint_dir, config=config)
        elif 'roberta-base' == args.model_type:
            model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    except:
        dirs = os.listdir(args.checkpoint_dir)
        dirs = [x for x in dirs if re.search('.*_best_checkpoint', x)]
        assert len(dirs) == 1
        checkpoint_dir = os.path.join(args.checkpoint_dir, dirs[0])
        config = AutoConfig.from_pretrained(checkpoint_dir)
        if 'roberta-base-custom' == args.model_type:
            model = utils.RobertaCustomForSequenceClassification.from_pretrained(checkpoint_dir, config=config)
        elif 'roberta-base' == args.model_type:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, config=config)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
    model.to(device)

    if args.test_data_path.endswith('.tsv'):
        test_data = pd.read_csv(args.test_data_path, sep='\t')
    elif args.test_data_path.endswith('.jsonl'):
        test_data = pd.read_json(args.test_data_path, lines=True)

    test_dataset = Dataset.from_pandas(test_data)

    def collate_fn(batch):
        outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
        outputs['labels'] = torch.Tensor([[1-example['labels'], example['labels']] for example in batch])
        return outputs
    
    test_loader = DataLoader(test_dataset, shuffle=False,batch_size=args.batch_size,collate_fn=collate_fn)

    predictions = []
    for batch in test_loader:
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'].to(device), 
                            labels = batch['labels'].to(device),
                            attention_mask = batch['attention_mask'].to(device))
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions.extend(probs[:, 1].reshape(-1,).tolist())


    labels_predictions = pd.DataFrame({"predictions": predictions, "labels": test_data['labels'].tolist()})
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

    # Create a confusion matrix
    labels_predictions['pred_cat']=pd.cut(labels_predictions['predictions'], np.arange(0, 1.1, 0.25), include_lowest=True, right=True)
    labels_predictions['labels_cat']=pd.cut(labels_predictions['labels'], np.arange(0, 1.1, 0.25), include_lowest=True, right=True)        
  
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_type", type=str, choices=['roberta-base', 'roberta-base-custom'])
    args = parser.parse_args()

    main(args)

