#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import argparse
import re
import json


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
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, f1_score
import seaborn as sns

import mucoco.utils as utils
from new_module.em_training.nli.models import EncoderModel
from new_module.utils.load_ckpt import define_model


def predict_labels(args, device):
    
    if 'task' not in args:
        args.task = ''
        
    if 'label_id' not in args:
        args.label_id = 1
        
    if 'embedding_model' not in args:
        args.embedding_model = ''
    
    if 'encoder_model' not in args:
        args.encoder_model = ''
    
    if args.model_type == 'encoder-model':
        with open(os.path.join(args.checkpoint_dir, 'config.json')) as f:
            model_config = json.load(f)
        model_config['device'] = device
        model_config['model_path'] = os.path.join(args.checkpoint_dir, args.model_file_name)
        
        model = EncoderModel(params=model_config)
        model.load_state_dict(torch.load(model_config['model_path'],weights_only=True),strict=False)

        tokenizer = model.tokenizer
    elif ('custom' in args.model_type) and (args.task == 'nli'):
        
        model, tokenizer = define_model(
            num_classes=2, 
            mod_path=os.path.join(args.checkpoint_dir, args.model_file_name),
            load_weights=True,
            device=device,
            embedding_model=args.embedding_model,
            encoder_model=args.encoder_model,
            task='nli'
        )
        
    else:
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

    if args.task == 'nli':
        test_data = test_data.loc[test_data['split']=='dev'].copy()
        test_data = test_data.rename(columns={'finegrained_labels': 'labels'})
    test_dataset = Dataset.from_pandas(test_data)

    if args.task == "nli":
        def collate_fn(batch):
            premises = [example['premise'] for example in batch]
            hypotheses =[example['hypothesis'] for example in batch] 
            sequences = [tokenizer.bos_token + p + tokenizer.sep_token + h + tokenizer.eos_token for p,h in zip(premises,hypotheses)]
            outputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
            return outputs
    else:
        def collate_fn(batch):
            outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
            return outputs
    
    test_loader = DataLoader(test_dataset, shuffle=False,batch_size=args.batch_size,collate_fn=collate_fn)

    predictions = []
    for batch in test_loader:
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'].to(device), 
                            attention_mask = batch['attention_mask'].to(device))
            if args.model_type == 'encoder-model':
                probs = torch.softmax(outputs[0], dim=-1)
            else:
                probs = torch.softmax(outputs.logits, dim=-1)
            predictions.extend(probs[:, args.label_id].reshape(-1,).tolist())

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


    ### Obtain binary labels
    labels_predictions['labels_binary'] = labels_predictions['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    labels_predictions['predictions_binary'] = labels_predictions['predictions'].apply(lambda x: 1 if x > 0.5 else 0)

    ### Plot & Analyze Model Outputs

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Classification Accuracy: {accuracy_score(labels_predictions['labels_binary'], labels_predictions['predictions_binary'])}\n")
        f.write(f"Classification F1: {f1_score(labels_predictions['labels_binary'], labels_predictions['predictions_binary'])}\n")
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
    parser.add_argument("--model_type", type=str, choices=['roberta-base', 'roberta-base-custom', 'encoder-model'])
    parser.add_argument("--model_file_name", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    main(args)

