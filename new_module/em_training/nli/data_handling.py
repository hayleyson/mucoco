import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, BatchSampler, SequentialSampler
import matplotlib.pyplot as plt
import pandas as pd
import json
import math
    
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
   
    # def __getitem__(self, idx):
    #     return idx

    def set_label_column(self, label_column):
        self.label_column = label_column
    
class NLI_BatchSampler(Sampler):
    
    def __init__(self, examples, batch_size, shuffle=True, allow_oversample = True):
        self.shuffle = shuffle
        self.drop_last = True ## drop_last = False is not implemented
        self.batch_size = batch_size
        self.allow_oversample = allow_oversample
        self.data = {'nan': [],
                     '0': [],
                     '(0, 0.5)': [],
                     '[0.5, 1)': [],
                     '1': []}
        for idx, item in enumerate(examples.to_dict(orient='records')):
            if math.isnan(item['finegrained_labels']):
                self.data['nan'].append(idx)
            elif item['finegrained_labels'] == 0.0:
                self.data['0'].append(idx)
            elif item['finegrained_labels'] == 1.0:
                self.data['1'].append(idx)
            elif (item['finegrained_labels'] > 0.0) and (item['finegrained_labels'] < 0.5):
                self.data['(0, 0.5)'].append(idx)
            else:
                self.data['[0.5, 1)'].append(idx)
        
        for key in self.data.keys():
            self.data[key] = torch.LongTensor(self.data[key])
        
        # each batch of finegrained data will be composed of 2:1:1:2 ratios of (0, 0~0.5, 0.5~1, 1) data
        self.fg_inbatch_cnt_05_1 = self.fg_inbatch_cnt_0_05 = self.batch_size // 6
        self.fg_inbatch_cnt_1 = (self.batch_size - 2 * self.fg_inbatch_cnt_05_1) // 2
        self.fg_inbatch_cnt_0 = self.batch_size - self.fg_inbatch_cnt_05_1 - self.fg_inbatch_cnt_0_05 - self.fg_inbatch_cnt_1
        
        if self.allow_oversample:
            self.num_finegrained_batches = max(len(self.data['1']) // self.fg_inbatch_cnt_1, 
                                        max(len(self.data['0']) // self.fg_inbatch_cnt_0,
                                            max(len(self.data['[0.5, 1)']) // self.fg_inbatch_cnt_05_1, 
                                                len(self.data['(0, 0.5)']) // self.fg_inbatch_cnt_0_05
                                                )
                                        )
                          )
        else:
            self.num_finegrained_batches = min(len(self.data['1']) // self.fg_inbatch_cnt_1, 
                                        min(len(self.data['0']) // self.fg_inbatch_cnt_0,
                                            min(len(self.data['[0.5, 1)']) // self.fg_inbatch_cnt_05_1, 
                                                len(self.data['(0, 0.5)']) // self.fg_inbatch_cnt_0_05
                                                )
                                        )
                          )
        self.num_nan_batches = (len(self.data['nan']) // self.batch_size)
        
        self.num_batches = self.num_nan_batches + self.num_finegrained_batches
        
        if not self.shuffle:
            # extend indexes if some bin of data have to be oversampled
            self.data['0'] = self.data['0'].repeat(math.ceil((self.fg_inbatch_cnt_0 * self.num_finegrained_batches) / len(self.data['0'])))
            self.data['0'] = self.data['0'][:self.fg_inbatch_cnt_0 * self.num_finegrained_batches]
            
            self.data['1'] = self.data['1'].repeat(math.ceil((self.fg_inbatch_cnt_1 * self.num_finegrained_batches) / len(self.data['1'])))
            self.data['1'] = self.data['1'][:self.fg_inbatch_cnt_1 * self.num_finegrained_batches]
            
            self.data['(0, 0.5)'] = self.data['(0, 0.5)'].repeat(math.ceil((self.fg_inbatch_cnt_0_05 * self.num_finegrained_batches) / len(self.data['(0, 0.5)'])))
            self.data['(0, 0.5)'] = self.data['(0, 0.5)'][:self.fg_inbatch_cnt_0_05 * self.num_finegrained_batches]
            
            self.data['[0.5, 1)'] = self.data['[0.5, 1)'].repeat(math.ceil((self.fg_inbatch_cnt_05_1 * self.num_finegrained_batches) / len(self.data['[0.5, 1)'])))
            self.data['[0.5, 1)'] = self.data['[0.5, 1)'][:self.fg_inbatch_cnt_05_1 * self.num_finegrained_batches]
            
    
    def __iter__(self):
        
        # set samplers for each type of data
        if self.shuffle:
            self.rsampler_nan = BatchSampler(RandomSampler(self.data['nan']), self.batch_size, self.drop_last)
            self.rsampler_0 = BatchSampler(RandomSampler(self.data['0'], num_samples = self.num_finegrained_batches * self.fg_inbatch_cnt_0),
                                        batch_size=self.fg_inbatch_cnt_0, drop_last = True)
            self.rsampler_1 = BatchSampler(RandomSampler(self.data['1'], num_samples = self.num_finegrained_batches * self.fg_inbatch_cnt_1),
                                        batch_size=self.fg_inbatch_cnt_1, drop_last = True)
            self.rsampler_0_05 = BatchSampler(RandomSampler(self.data['(0, 0.5)'], num_samples = self.num_finegrained_batches * self.fg_inbatch_cnt_0_05),
                                        batch_size=self.fg_inbatch_cnt_0_05, drop_last = True)
            self.rsampler_05_1 = BatchSampler(RandomSampler(self.data['[0.5, 1)'], num_samples = self.num_finegrained_batches * self.fg_inbatch_cnt_05_1),
                                        batch_size=self.fg_inbatch_cnt_05_1, drop_last = True)
        else:
            self.rsampler_nan = BatchSampler(SequentialSampler(self.data['nan']), self.batch_size, self.drop_last)
            self.rsampler_0 = BatchSampler(SequentialSampler(self.data['0']), batch_size=self.fg_inbatch_cnt_0, drop_last = True)
            self.rsampler_1 = BatchSampler(SequentialSampler(self.data['1']), batch_size=self.fg_inbatch_cnt_1, drop_last = True)
            self.rsampler_0_05 = BatchSampler(SequentialSampler(self.data['(0, 0.5)']), batch_size=self.fg_inbatch_cnt_0_05, drop_last = True)
            self.rsampler_05_1 = BatchSampler(SequentialSampler(self.data['[0.5, 1)']), batch_size=self.fg_inbatch_cnt_05_1, drop_last = True)
    
        self.rsampler_0 = iter(self.rsampler_0)
        self.rsampler_1 = iter(self.rsampler_1)
        self.rsampler_nan = iter(self.rsampler_nan)
        self.rsampler_0_05 = iter(self.rsampler_0_05)
        self.rsampler_05_1 = iter(self.rsampler_05_1)
        
        batch = []
        # use up samples with only discrete labels first
        for batch in self.rsampler_nan:
            yield self.data['nan'][batch].tolist()
            batch = []
                
        # use samples with finegrained labels
        for _ in range(self.num_finegrained_batches):
            batch.extend(self.data['0'][next(self.rsampler_0)].tolist())
            batch.extend(self.data['(0, 0.5)'][next(self.rsampler_0_05)].tolist())
            batch.extend(self.data['[0.5, 1)'][next(self.rsampler_05_1)].tolist())
            batch.extend(self.data['1'][next(self.rsampler_1)].tolist())
            yield batch
            batch = []
        
    def __len__(self):
        return self.num_batches
                
    
class NLI_DataLoader:
    
    def __init__(self, dataset, config, mode, tokenizer, allow_oversample=True):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.batch_size = self.config['energynet']['batch_size']
        self.mode = mode
        self.shuffle = True if self.mode=='train' else False
        self.allow_oversample = allow_oversample
        self.batch_sampler = self.get_batch_sampler()
    
    # def collate_fn(self, batch):
    #     return batch
    
    def collate_fn(self, batch):
        premises = [x[0] for x in batch]
        hypotheses = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        finegrained_labels = [x[3] for x in batch]
        
        sequences = [self.tokenizer.bos_token + p + self.tokenizer.sep_token + h + self.tokenizer.eos_token for p, h in zip(premises, hypotheses)]
        tokenized_sequences = self.tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
        if (self.config['energynet']['loss'] == 'cross_entropy') and (self.config['energynet']['label_column'] == 'finegrained_labels'):
            labels = torch.Tensor(labels).reshape(-1, 1)
            labels = torch.tile(labels, (1,2))
            labels[:, 0] = 1 - labels[:, 0] 
        elif (self.config['energynet']['loss'] == 'binary_cross_entropy') or ((self.config['energynet']['loss'] == 'cross_entropy') and (self.config['energynet']['label_column'] == 'original_labels')):
            labels = torch.LongTensor(labels)
        elif self.config['energynet']['loss'] in ['mse', 'margin_ranking', 'scaled_ranking', 'mse+margin_ranking']:
            labels = torch.Tensor(labels).reshape(-1, 1)
        else:
            raise NotImplementedError('Loss type not recognized')
        
        return {'input_ids': tokenized_sequences['input_ids'].to(self.config['device']), 
                'attention_mask': tokenized_sequences['attention_mask'].to(self.config['device']), 
                'labels': labels.to(self.config['device']),
                'finegrained_labels': torch.Tensor(finegrained_labels).to(self.config['device'])}
    
    def get_batch_sampler(self):
        return NLI_BatchSampler(self.dataset.data, self.batch_size, shuffle=self.shuffle, allow_oversample=self.allow_oversample)
    
    def get_dataloader(self):
        if self.shuffle:
            return DataLoader(self.dataset, batch_sampler = self.batch_sampler, collate_fn=self.collate_fn)
        else:
            return DataLoader(self.dataset, batch_size = self.batch_size, shuffle = self.shuffle, collate_fn=self.collate_fn)
        
    
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
        
        # change column name 'label' to 'binary_labels'
        nli_dataset = nli_dataset.rename(columns={'label': 'binary_labels'})
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
        
        # change column name 'label' to 'binary_labels'
        nli_dataset = nli_dataset.rename(columns={'label': 'binary_labels'})
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
        # change column name 'label' to 'binary_labels'
        nli_dataset = nli_dataset.rename(columns={'label': 'binary_labels'})
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