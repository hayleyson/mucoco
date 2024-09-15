import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import json
    
class NLI_Dataset(Dataset):
    
    def __init__(self, dataframe, label_column):
        self.data = dataframe
        self.label_column = None
        self.set_label_column(label_column)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        if 'finegrained_labels' in self.data.columns:
            return (sample['premise'], sample['hypothesis'], sample[self.label_column], sample['finegrained_labels'])
        else:
            return (sample['premise'], sample['hypothesis'], sample[self.label_column], [None for _ in range(len(sample))])

    def set_label_column(self, label_column):
        self.label_column = label_column
    
class NLI_DataLoader:
    
    def __init__(self, dataset, config, mode, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.batch_size = self.config['energynet']['batch_size']
        self.mode = mode
    
    def collate_fn(self, batch):
        premises = [x[0] for x in batch]
        hypotheses = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        finegrained_labels = [x[3] for x in batch]
        
        sequences = [self.tokenizer.bos_token + p + self.tokenizer.sep_token + h + self.tokenizer.eos_token for p, h in zip(premises, hypotheses)]
        tokenized_sequences = self.tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
        if self.config['energynet']['loss'] == 'cross_entropy':
            labels = torch.Tensor(labels).reshape(-1, 1)
            labels = torch.tile(labels, (1,2))
            labels[:, 0] = 1 - labels[:, 0] 
        elif self.config['energynet']['loss'] == 'binary_cross_entropy':
            labels = torch.LongTensor(labels)
        elif self.config['energynet']['loss'] in ['mse', 'margin_ranking', 'negative_log_odds', 'mse+margin_ranking']:
            labels = torch.Tensor(labels).reshape(-1, 1)
        else:
            raise NotImplementedError('Loss type not recognized')
        
        return {'input_ids': tokenized_sequences['input_ids'].to(self.config['device']), 
                'attention_mask': tokenized_sequences['attention_mask'].to(self.config['device']), 
                'labels': labels.to(self.config['device']),
                'finegrained_labels': finegrained_labels}
    
    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True if self.mode=='train' else False, collate_fn=self.collate_fn)
        
    
def load_nli_data(dev_split_size=0.1, force_reload=False,
                  output_file_path = 'data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl'):
    
    """
    Load and preprocess SNLI, MNLI, ANLI datasets
    1. Load data
    2. Preprocess SNLI, MNLI datasets 
        - concat data
        - drop samples with '-' label
        - add finegrained labels
        - split data by promptID considering genre and source
    3. Preprocess ANLI dataset
        - concat data
        - add finegrained labels
        - split data considering genre and source
    4. Concat SNLI, MNLI, ANLI data and save
    5. Save dataset statistics
    """
        
    if os.path.exists(output_file_path) and not force_reload:
        print('Preprocessed dataset already exists. Loading data from file')
        data = pd.read_json(output_file_path, lines=True)
        return data
    else:
        
        # 1. Load data
        mnli1 = pd.read_json('data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl', lines=True)
        mnli2 = pd.read_json('data/nli/multinli_1.0/multinli_1.0_dev_mismatched.jsonl', lines=True)
        snli = pd.read_json('data/nli/snli_1.0/snli_1.0_dev.jsonl', lines=True).rename(columns={'captionID': 'promptID'})
        anli_R1_train = pd.read_json('data/nli/anli_v1.0/R1/train.jsonl', lines=True)
        anli_R1_dev = pd.read_json('data/nli/anli_v1.0/R1/dev.jsonl', lines=True)
        anli_R2_train = pd.read_json('data/nli/anli_v1.0/R2/train.jsonl', lines=True)
        anli_R2_dev = pd.read_json('data/nli/anli_v1.0/R2/dev.jsonl', lines=True)
        anli_R3_train = pd.read_json('data/nli/anli_v1.0/R3/train.jsonl', lines=True)
        anli_R3_dev = pd.read_json('data/nli/anli_v1.0/R3/dev.jsonl', lines=True)
        
        # 2. Preprocess SNLI, MNLI datasets 
        # 2-1. concat data
        smnli = pd.concat([mnli1,mnli2,snli],axis=0)
        smnli['genre'].fillna('caption', inplace=True)
        smnli = smnli.rename(columns={'gold_label': 'label',
                                    'sentence1': 'premise',
                                    'sentence2': 'hypothesis'})
        smnli['source'] = ['mnli_matched']*len(mnli1) + ['mnli_mismatched']*len(mnli2) + ['snli']*len(snli)
        
        # 2-2. drop samples with '-' label
        smnli = smnli.loc[smnli['label'] != '-', :].copy()
        assert smnli.shape[0] == 29489, "Number of samples in SMNLI data is not 29489"
        
        # 2-3. add finegrained labels 
        smnli['finegrained_labels'] = smnli['annotator_labels'].apply(lambda labels: np.mean([1 if label == 'contradiction' else 0 for label in labels]))
        smnli['original_labels'] = smnli['label'].replace({'entailment': 0, 'neutral': 1, 'contradiction': 2})
        smnli['label'] = smnli['label'].apply(lambda x: 1 if x == 'contradiction' else 0)
        
        
        # 2-4. split data by promptID considering genre and source
        smnli_info_for_sample = smnli[['promptID', 'genre', 'source']].drop_duplicates()    
        train, dev = train_test_split(smnli_info_for_sample['promptID'], test_size=dev_split_size, random_state=42, stratify=smnli_info_for_sample[['genre','source']])
        smnli['split'] = ['train' if x in train.values else 'dev' for x in smnli['promptID']]
        
        smnli = smnli[['premise', 'hypothesis', 'original_labels', 'label', 'finegrained_labels', 'genre', 'source', 'split']]
        
        # 3. Preprocess ANLI dataset
        # 3-1. concat data
        anli = pd.concat([anli_R1_train, anli_R1_dev, anli_R2_train, anli_R2_dev, anli_R3_train, anli_R3_dev],axis=0)
        anli = anli.rename(columns={'context': 'premise'})
        anli['source'] = ['anli_R1_train']*len(anli_R1_train) + ['anli_R1_dev']*len(anli_R1_dev) + ['anli_R2_train']*len(anli_R2_train) + ['anli_R2_dev']*len(anli_R2_dev) + ['anli_R3_train']*len(anli_R3_train) + ['anli_R3_dev']*len(anli_R3_dev)
        
        # 3-2. add finegrained labels
        anli_verifier_labels = pd.read_json('data/nli/anli_v1.0/verifier_labels_R1-3.jsonl', lines=True)
        anli = pd.merge(anli,anli_verifier_labels, on='uid', how='inner')
        assert anli.shape[0] == 26603, "Number of samples in ANLI data is not 26603"
        anli['finegrained_labels'] = anli['verifier labels'].apply(lambda labels: np.mean([1 if label == 'c' else 0 for label in labels]))
        anli['original_labels'] = anli['label'].replace({'e': 0, 'n': 1, 'c': 2})
        anli['label'] = anli['label'].apply(lambda x: 1 if x == 'c' else 0)
        
        # 3-3. split data considering genre and source
        # anli['split'] = ['train' if 'train' in x else 'dev' for x in anli['source']]
        anli = anli.reset_index(drop=True)
        train, dev = train_test_split(anli, test_size=dev_split_size, random_state=42, stratify=anli[['genre','source']])
        anli['split'] = ['train' if x in train.index.tolist() else 'dev' for x in anli.index]
        
        anli = anli[['premise', 'hypothesis', 'original_labels', 'label', 'finegrained_labels', 'genre', 'source', 'split' ]]
        
        # 4. Concat SNLI, MNLI, ANLI data and save
        nli_dataset = pd.concat([smnli, anli], axis=0)
        nli_dataset.to_json(output_file_path, orient='records', lines=True)

        # 5. Save dataset statistics
        f = open(output_file_path.replace('.jsonl', '_stats.txt'), 'w')
        
        print("Number of samples by source", file=f)
        nli_dataset['source_abb'] = nli_dataset['source'].apply(lambda x: 'mnli' if 'mnli' in x else 'snli' if 'snli' in x else 'anli')
        print(nli_dataset.groupby(['source_abb']).size(), file=f)
        print(file=f)
        
        print("Number of samples by split", file=f)
        print(nli_dataset['split'].value_counts(), file=f)
        print(file=f)

        print("Distribution of labels", file=f)
        bins = pd.cut(nli_dataset['finegrained_labels'], np.arange(-0.1, 1.1, 0.1))
        print(bins.value_counts().sort_index(), file=f)
        f.close()
        
        nli_dataset['finegrained_labels'].hist(bins=30)
        plt.savefig(output_file_path.replace('.jsonl', '_hist.png'))
        
        return nli_dataset
    
def load_additional_nli_training_data(force_reload=False,
                                      output_file_path = 'data/nli/snli_mnli_anli_train_without_finegrained.jsonl'):
    if os.path.exists(output_file_path) and not force_reload:
        print('Preprocessed dataset already exists. Loading data from file')
        data = pd.read_json(output_file_path, lines=True)
        return data
    else:
        
        # 1. Load data
        mnli = pd.read_json('data/nli/multinli_1.0/multinli_1.0_train.jsonl', lines=True)
        snli = pd.read_json('data/nli/snli_1.0/snli_1.0_train.jsonl', lines=True).rename(columns={'captionID': 'promptID'})
        anli_R1_train = pd.read_json('data/nli/anli_v1.0/R1/train.jsonl', lines=True)
        anli_R2_train = pd.read_json('data/nli/anli_v1.0/R2/train.jsonl', lines=True)
        anli_R3_train = pd.read_json('data/nli/anli_v1.0/R3/train.jsonl', lines=True)
        
        # 2. Preprocess SNLI, MNLI datasets 
        # 2-1. concat data
        smnli = pd.concat([mnli,snli],axis=0)
        smnli['genre'].fillna('caption', inplace=True)
        smnli = smnli.rename(columns={'gold_label': 'label',
                                    'sentence1': 'premise',
                                    'sentence2': 'hypothesis'})
        smnli['source'] = ['mnli']*len(mnli) + ['snli']*len(snli)
        
        # 2-2. drop samples with '-' label
        smnli = smnli.loc[smnli['label'] != '-', :].copy()
        assert smnli.shape[0] == 942069, f"Number of samples {smnli.shape[0]} in SMNLI data is not 689614"
        
        # 2-3. add labels & split
        smnli['finegrained_labels'] = [None for _ in range(len(smnli))]
        smnli['original_labels'] = smnli['label'].replace({'entailment': 0, 'neutral': 1, 'contradiction': 2})
        smnli['label'] = smnli['label'].apply(lambda x: 1 if x == 'contradiction' else 0)
        smnli['split'] = ['train' for _ in range(len(smnli))]
        smnli = smnli[['premise', 'hypothesis', 'original_labels', 'label', 'finegrained_labels', 'genre', 'source', 'split']]
        
        # 3. Preprocess ANLI dataset
        # 3-1. concat data
        anli = pd.concat([anli_R1_train, anli_R2_train, anli_R3_train],axis=0)
        anli = anli.rename(columns={'context': 'premise'})
        anli['source'] = ['anli_R1_train']*len(anli_R1_train)  + ['anli_R2_train']*len(anli_R2_train) + ['anli_R3_train']*len(anli_R3_train)
        
        # 3-2. add labels
        # only select rows without finegrained labels
        anli_verifier_labels = pd.read_json('data/nli/anli_v1.0/verifier_labels_R1-3.jsonl', lines=True)
        anli = anli.loc[~anli['uid'].isin(anli_verifier_labels['uid'].tolist())]
        
        anli['finegrained_labels'] = [None for _ in range(len(anli))]
        anli['original_labels'] = anli['label'].replace({'e': 0, 'n': 1, 'c': 2})
        anli['label'] = anli['label'].apply(lambda x: 1 if x == 'c' else 0)
        
        # 3-3. add split info
        anli['split'] = ['train' for _ in range(len(anli))]
        anli = anli[['premise', 'hypothesis', 'original_labels', 'label', 'finegrained_labels', 'genre', 'source', 'split' ]]
        
        # 4. Concat SNLI, MNLI, ANLI data and save
        nli_dataset = pd.concat([smnli, anli], axis=0)
        nli_dataset.to_json(output_file_path, orient='records', lines=True)

        # 5. Save dataset statistics
        f = open(output_file_path.replace('.jsonl', '_stats.txt'), 'w')
        
        print("Number of samples by source", file=f)
        nli_dataset['source_abb'] = nli_dataset['source'].apply(lambda x: 'mnli' if 'mnli' in x else 'snli' if 'snli' in x else 'anli')
        print(nli_dataset.groupby(['source_abb']).size(), file=f)
        print(file=f)
        
        print("Number of samples by split", file=f)
        print(nli_dataset['split'].value_counts(), file=f)
        print(file=f)
        
        return nli_dataset
    
        
def load_nli_test_data(force_reload=False,
                        output_file_path = 'data/nli/snli_mnli_anli_test_with_finegrained.jsonl'):
    
    """
    Load and preprocess SNLI, MNLI, ANLI **test** datasets 
    1. Load data
    2. Preprocess SNLI, MNLI datasets 
        - concat data
        - drop samples with '-' label
        - add finegrained labels
    3. Preprocess ANLI dataset
        - concat data
        - add finegrained labels
    4. Concat SNLI, MNLI, ANLI data and save
    5. Save dataset statistics
    """
    if os.path.exists(output_file_path) and not force_reload:
        print('Preprocessed dataset already exists. Loading data from file')
        data = pd.read_json(output_file_path, lines=True)
        return data
    else:
        
        # 1. Load data
        # no mnli test data
        snli = pd.read_json('data/nli/snli_1.0/snli_1.0_test.jsonl', lines=True).rename(columns={'captionID': 'promptID'})
        anli_R1_test = pd.read_json('data/nli/anli_v1.0/R1/test.jsonl', lines=True)
        anli_R2_test = pd.read_json('data/nli/anli_v1.0/R2/test.jsonl', lines=True)
        anli_R3_test = pd.read_json('data/nli/anli_v1.0/R3/test.jsonl', lines=True)
        
        # 2. Preprocess SNLI, MNLI datasets 
        # 2-1. concat data
        smnli = pd.concat([snli],axis=0)
        smnli['genre'] = ['caption'] * len(snli)
        smnli = smnli.rename(columns={'gold_label': 'label',
                                    'sentence1': 'premise',
                                    'sentence2': 'hypothesis'})
        smnli['source'] = ['snli']*len(snli)
        
        # 2-2. drop samples with '-' label
        smnli = smnli.loc[smnli['label'] != '-', :].copy()
        assert smnli.shape[0] == 9824, f"Number of samples in SMNLI data,  {len(smnli)}, is not 9824"
        
        # 2-3. add finegrained labels 
        smnli['finegrained_labels'] = smnli['annotator_labels'].apply(lambda labels: np.mean([1 if label == 'contradiction' else 0 for label in labels]))
        smnli['original_labels'] = smnli['label'].replace({'entailment': 0, 'neutral': 1, 'contradiction': 2})
        smnli['label'] = smnli['label'].apply(lambda x: 1 if x == 'contradiction' else 0)
        smnli = smnli[['premise', 'hypothesis', 'original_labels', 'label', 'finegrained_labels', 'genre', 'source']]
        
        # 3. Preprocess ANLI dataset
        # 3-1. concat data
        anli = pd.concat([anli_R1_test, anli_R2_test, anli_R3_test],axis=0)
        anli = anli.rename(columns={'context': 'premise'})
        anli['source'] = ['anli_R1_test']*len(anli_R1_test) + ['anli_R2_train']*len(anli_R2_test) + ['anli_R3_test']*len(anli_R3_test)
        
        # 3-2. add finegrained labels
        anli_verifier_labels = pd.read_json('data/nli/anli_v1.0/verifier_labels_R1-3.jsonl', lines=True)
        anli = pd.merge(anli,anli_verifier_labels, on='uid', how='inner')
        assert anli.shape[0] == 3200, "Number of samples in ANLI data is not 3200"
        anli['finegrained_labels'] = anli['verifier labels'].apply(lambda labels: np.mean([1 if label == 'c' else 0 for label in labels]))
        anli['original_labels'] = anli['label'].replace({'e': 0, 'n': 1, 'c': 2})
        anli['label'] = anli['label'].apply(lambda x: 1 if x == 'c' else 0)
        anli = anli[['premise', 'hypothesis', 'original_labels', 'label', 'finegrained_labels', 'genre', 'source']]
        
        # 4. Concat SNLI, MNLI, ANLI data and save
        nli_dataset = pd.concat([smnli, anli], axis=0)
        nli_dataset.to_json(output_file_path, orient='records', lines=True)

        # 5. Save dataset statistics
        f = open(output_file_path.replace('.jsonl', '_stats.txt'), 'w')
        
        print("Number of samples by source", file=f)
        nli_dataset['source_abb'] = nli_dataset['source'].apply(lambda x: 'mnli' if 'mnli' in x else 'snli' if 'snli' in x else 'anli')
        print(nli_dataset.groupby(['source_abb']).size(), file=f)
        print(file=f)
        

        print("Distribution of labels", file=f)
        bins = pd.cut(nli_dataset['finegrained_labels'], np.arange(-0.1, 1.1, 0.1))
        print(bins.value_counts().sort_index(), file=f)
        f.close()
        
        nli_dataset['finegrained_labels'].hist(bins=30)
        plt.savefig(output_file_path.replace('.jsonl', '_hist.png'))
        
        return nli_dataset
    
    


if __name__ == "__main__":
    # load_nli_data(0.1,force_reload=True)
    # load_nli_test_data(force_reload=True)
    load_additional_nli_training_data(force_reload=True)