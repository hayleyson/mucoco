"""
This script finds the classification threshold that achieves the best [metric] score on the validation dataset
and saves the testset edit candidates according to the best F1 threshold.
It can also be run with a specific threshold to save the testset edit candidates.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import json
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support
import torch.nn.functional as F
from new_module.ebm_training.nli.models.encoder import EncoderModel
from new_module.utils.utils import read_outputs, ravel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# MEMO
# - toxicity valid dataset: data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/valid.jsonl
# - nli valid dataset: data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl

def load_model_and_tokenizer(model_path, device, task):
    if task == "toxicity":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif task == "nli":
        # config
        try:
            with open(os.path.join(model_path, 'config.json')) as f:
                model_config = json.load(f)
        except:
            model_config = {'energynet': {'base_model': 'roberta-large',
                                          'output_form': '2dim_vec'},
                            'locate': {'type': ''}}
        model_config['device'] = device
        model_config['model_path'] = os.path.join(model_path, 'best_model_pearsonr.pth')
        model_config['locate']['type'] = "gradnorm"
        model = EncoderModel(params=model_config)
        model.load_state_dict(torch.load(model_config['model_path'],weights_only=True),strict=False)
        model.eval()
        model.to(device)
        # tokenizer
        tokenizer = model.tokenizer
        
    else:
        raise ValueError(f"Invalid task: {task}")
    return model, tokenizer


def load_dataset(dataset_path, task):
    """
    Load the dataset from the given path.
    """
    if task == 'toxicity':
        data = pd.read_json(dataset_path, lines=True)    
        data['labels'] = data['labels'].apply(lambda x: 1 if x <= 0.5 else 0)    
    elif task == 'nli':
        data = pd.read_json(dataset_path, lines=True)
        data = data[['pairID', 'premise', 'hypothesis', 'binary_labels', 'split']]
        data = data[data['split'] == 'dev']
        data = data.rename(columns={'binary_labels': 'labels'})
        del data['split']
    else:
        raise ValueError(f"Invalid task: {task}")
    print(f"Class distribution in validation dataset: {data['labels'].value_counts()}")
    return Dataset.from_pandas(data)


def find_classification_threshold(model_path, validation_dataset_path, task, label_id, batch_size=64, num_workers=2):
    """
    Find the classification threshold that achieves the best F1 score on the validation dataset.
    """
        
    # load model
    model, tokenizer = load_model_and_tokenizer(model_path, device, task)
    
    dataset = load_dataset(validation_dataset_path, task)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    energy_vals = []
    for batch in dataloader:
        if task == "toxicity":
            with torch.no_grad():
                inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to(device)
                outputs = model(**inputs)
                logits = outputs.logits
        elif task == "nli":
            with torch.no_grad():
                inputs = tokenizer([tokenizer.bos_token + p + tokenizer.sep_token + h + tokenizer.eos_token for p, h in zip(batch['premise'], batch['hypothesis'])], return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to(device)
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                logits = outputs[0]
        else:
            raise ValueError(f"Invalid task: {task}")   
        # print(f"logits shape: {logits.shape}")
        energy_vals.append(-F.log_softmax(logits, dim=-1).detach().cpu().numpy()[:, label_id])
    
    
    energy_vals = np.concatenate(energy_vals, axis=0)
    candidate_thresholds = np.unique(energy_vals) # sorted in ascending order
    best_f1, best_acc, best_recall, best_precision = 0, 0, 0, 0
    best_f1_threshold_list, best_acc_threshold_list, best_recall_threshold_list, best_precision_threshold_list = [], [], [], []
    for threshold in candidate_thresholds:
        labels = (energy_vals < threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(dataset['labels'], labels, average='macro')
        accuracy = accuracy_score(dataset['labels'], labels)
        # estimating precision
        if best_precision < precision:
            best_precision = precision
            best_precision_threshold_list = [threshold]
        elif best_precision == precision:
            best_precision_threshold_list.append(threshold)

        # estimating recall
        if best_recall < recall:
            best_recall = recall
            best_recall_threshold_list = [threshold]
        elif best_recall == recall:
            best_recall_threshold_list.append(threshold)

        # estimating f1
        if best_f1 < f1:
            best_f1 = f1
            best_f1_threshold_list = [threshold]
        elif best_f1 == f1:
            best_f1_threshold_list.append(threshold)

        # estimating acc
        if best_acc < accuracy:
            best_acc = accuracy
            best_acc_threshold_list = [threshold]
        elif best_acc == accuracy:
            best_acc_threshold_list.append(threshold)
            
    return {
        'precision': float(best_precision),
        'precision_threshold': float(np.exp(-min(best_precision_threshold_list))),
        'recall': float(best_recall),
        'recall_threshold': float(np.exp(-max(best_recall_threshold_list))),
        'f1': float(best_f1),
        'f1_threshold': float(np.exp(-sum(best_f1_threshold_list) / len(best_f1_threshold_list))),
        'acc': float(best_acc),
        'acc_threshold': float(np.exp(-sum(best_acc_threshold_list) / len(best_acc_threshold_list))),
        'class_probability_percentiles': np.percentile(
            np.exp(-energy_vals), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ).astype(float).tolist(),
    }

def save_testset_edit_candidates(model_path, testset_path, task, label_id, threshold, batch_size=64, num_workers=2):
    if task not in ["toxicity", "nli"]:
        raise ValueError(f"Invalid task: {task}")
    model, tokenizer = load_model_and_tokenizer(model_path, device, task)
    data = read_outputs(testset_path)
    # read_outputs adds list-valued columns (e.g. prompt_instruction_id_list). Passing the
    # full frame into HuggingFace Dataset + default DataLoader collate makes PyTorch try
    # to tensor-stack those lists and raises: "each element in list of batch should be
    # of equal size". Only keep columns needed for the forward pass.
    if task == "nli":
        if "gen_text" in data.columns:
            infer_df = data.loc[:, ["prompt", "gen_text"]].rename(columns={"gen_text": "text"})
        elif "text" in data.columns:
            infer_df = data.loc[:, ["prompt", "text"]]
        else:
            raise ValueError("Expected gen_text or text column after read_outputs for nli")
    else:
        if "text" in data.columns:
            infer_df = data.loc[:, ["text"]]
        elif "gen_text" in data.columns:
            infer_df = data.loc[:, ["gen_text"]].rename(columns={"gen_text": "text"})
        else:
            raise ValueError("Expected text or gen_text column after read_outputs for toxicity")
    dataset = Dataset.from_pandas(infer_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    energy_vals = []
    for batch in dataloader:
        if task == "toxicity":
            with torch.no_grad():
                inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                energy_vals.append(-F.log_softmax(logits, dim=-1).detach().cpu().numpy()[:, label_id])
        elif task == "nli":
            with torch.no_grad():
                inputs = tokenizer([tokenizer.bos_token + p + tokenizer.sep_token + h + tokenizer.eos_token for p, h in zip(batch['prompt'], batch['text'])], return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--validation_dataset_path", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--label_id", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_testset_edit_candidates", action="store_true")
    parser.add_argument("--testset_path", type=str)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if args.threshold is None:
        result = find_classification_threshold(args.model_path, args.validation_dataset_path, args.task, args.label_id, args.batch_size, args.num_workers)
        with open(os.path.join(args.model_path, 'classification_threshold.json'), 'w') as f:
            f.write(json.dumps(result) + '\n')
        print(f"Best F1 threshold: {result['f1_threshold']}, Best F1: {result['f1']}")
        print(f"Best Accuracy threshold: {result['acc_threshold']}, Best Accuracy: {result['acc']}")
        print(f"Best Recall threshold: {result['recall_threshold']}, Best Recall: {result['recall']}")
        print(f"Best Precision threshold: {result['precision_threshold']}, Best Precision: {result['precision']}")
        print(f"Class probability percentiles: {result['class_probability_percentiles']}")
        if args.save_testset_edit_candidates:
            save_testset_edit_candidates(args.model_path, args.testset_path, args.task, args.label_id, result['f1_threshold'], args.batch_size, args.num_workers)
    else:
        if args.save_testset_edit_candidates:
            save_testset_edit_candidates(args.model_path, args.testset_path, args.task, args.label_id, args.threshold, args.batch_size, args.num_workers)