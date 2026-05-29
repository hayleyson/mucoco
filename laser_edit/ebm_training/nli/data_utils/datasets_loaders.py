import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, BatchSampler, SequentialSampler
import pandas as pd
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
                
class NLI_TrainBatchSampler_Continuous(Sampler):
    """
    Group examples by labels into intervals [0, 0], (0, 0.5), [0.5, 1), and [1, 1], 
    then sample examples from each bin in a 2:1:1:2 ratio
    """
    def __init__(self, examples, batch_size, oversample_minority = True):
        self.batch_size = batch_size
        self.oversample_minority = oversample_minority
        examples = examples.reset_index(drop=True)
        self.data = {'0': torch.LongTensor(examples.loc[examples['finegrained_labels'] == 0.0].index.tolist()),
                     '(0, 0.5)': torch.LongTensor(examples.loc[(examples['finegrained_labels'] > 0.0) & (examples['finegrained_labels'] < 0.5)].index.tolist()),
                     '[0.5, 1)': torch.LongTensor(examples.loc[(examples['finegrained_labels'] >= 0.5) & (examples['finegrained_labels'] < 1.0)].index.tolist()),
                     '1': torch.LongTensor(examples.loc[examples['finegrained_labels'] == 1.0].index.tolist())} ## 0: incon, 1: con
        
        # each batch of finegrained data will be composed of 2:1:1:2 ratios of (0, 0~0.5, 0.5~1, 1) data
        self.batch_size_05_1 = self.batch_size_0_05 = self.batch_size // 6
        self.batch_size_1 = (self.batch_size - 2 * self.batch_size_05_1) // 2
        self.batch_size_0 = self.batch_size - self.batch_size_05_1 - self.batch_size_0_05 - self.batch_size_1
        
        if self.oversample_minority:
            self.num_batches = max(len(self.data['1']) // self.batch_size_1, 
                                        max(len(self.data['0']) // self.batch_size_0,
                                            max(len(self.data['[0.5, 1)']) // self.batch_size_05_1, 
                                                len(self.data['(0, 0.5)']) // self.batch_size_0_05
                                                )
                                        )
                          )
        else:
            self.num_batches = min(len(self.data['1']) // self.batch_size_1, 
                                        min(len(self.data['0']) // self.batch_size_0,
                                            min(len(self.data['[0.5, 1)']) // self.batch_size_05_1, 
                                                len(self.data['(0, 0.5)']) // self.batch_size_0_05
                                                )
                                        )
                          )
    
    def __iter__(self):
        
        # set samplers for each type of data
        self.rsampler_0 = iter(BatchSampler(RandomSampler(self.data['0'], num_samples = self.num_batches * self.batch_size_0),
                                    batch_size=self.batch_size_0, drop_last = True))
        self.rsampler_1 = iter(BatchSampler(RandomSampler(self.data['1'], num_samples = self.num_batches * self.batch_size_1),
                                    batch_size=self.batch_size_1, drop_last = True))
        self.rsampler_0_05 = iter(BatchSampler(RandomSampler(self.data['(0, 0.5)'], num_samples = self.num_batches * self.batch_size_0_05),
                                    batch_size=self.batch_size_0_05, drop_last = True))
        self.rsampler_05_1 = iter(BatchSampler(RandomSampler(self.data['[0.5, 1)'], num_samples = self.num_batches * self.batch_size_05_1),
                                    batch_size=self.batch_size_05_1, drop_last = True))
        
        batch = []
                
        for _ in range(self.num_batches):
            batch.extend(self.data['0'][next(self.rsampler_0)].tolist())
            batch.extend(self.data['(0, 0.5)'][next(self.rsampler_0_05)].tolist())
            batch.extend(self.data['[0.5, 1)'][next(self.rsampler_05_1)].tolist())
            batch.extend(self.data['1'][next(self.rsampler_1)].tolist())
            yield batch
            batch = []
        
    def __len__(self):
        return self.num_batches
    
class NLI_TrainBatchSampler_Binary(Sampler):
    
    def __init__(self, examples:pd.DataFrame, batch_size, oversample_minority = True):
        self.batch_size = batch_size
        self.oversample_minority = oversample_minority
        
        examples = examples.reset_index(drop=True)
        self.data = {'con': torch.LongTensor(examples.loc[examples['binary_labels'] == 1.0].index.tolist()),
                     'incon': torch.LongTensor(examples.loc[examples['binary_labels'] == 0.0].index.tolist())}
        
        self.batch_size_con = self.batch_size - self.batch_size//2
        self.batch_size_incon = self.batch_size//2
        
        if self.oversample_minority:
            self.num_batches = max(len(self.data['con']) // self.batch_size_con, 
                                   len(self.data['incon']) // self.batch_size_incon)
        else:
            self.num_batches = min(len(self.data['con']) // self.batch_size_con, 
                                   len(self.data['incon']) // self.batch_size_incon)
            
    def __iter__(self):
        
        # set samplers for each type of data
        self.rsampler_incon = iter(BatchSampler(RandomSampler(self.data['incon'], 
                                    num_samples = self.num_batches * self.batch_size_incon),
                                    batch_size=self.batch_size_incon, drop_last = True))
        self.rsampler_con = iter(BatchSampler(RandomSampler(self.data['con'], 
                                    num_samples = self.num_batches * self.batch_size_con),
                                    batch_size=self.batch_size_con, drop_last = True))
        
        batch = []
        for _ in range(self.num_batches):
            batch.extend(self.data['incon'][next(self.rsampler_incon)].tolist())
            batch.extend(self.data['con'][next(self.rsampler_con)].tolist())
            yield batch
            batch = []
        
    def __len__(self):
        return self.num_batches
    
class NLI_DataLoader:
    
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = self.config['energynet']['batch_size']
    
    def collate_fn(self, batch):
        premises = [x[0] for x in batch]
        hypotheses = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        finegrained_labels = [x[3] for x in batch]

        if (self.config['energynet'].get('input_form') is None) or (self.config['energynet']['input_form'] == 'x_only'):  
            sequences = [self.tokenizer.bos_token + p + self.tokenizer.sep_token + h + self.tokenizer.eos_token for p, h in zip(premises, hypotheses)]
                
        elif self.config['energynet']['input_form'] == 'xy_concat':
            sequences_cons = [self.tokenizer.bos_token + p + self.tokenizer.sep_token + h + self.tokenizer.sep_token + "consistent" + self.tokenizer.eos_token for p, h in zip(premises, hypotheses)]
            sequences_incons = [self.tokenizer.bos_token + p + self.tokenizer.sep_token + h + self.tokenizer.sep_token + "inconsistent" + self.tokenizer.eos_token for p, h in zip(premises, hypotheses)]
            sequences = sequences_cons + sequences_incons
            
            ## for sequence_incons, labels need to be flipped (b/c current labels are proba of consistency or class 1 indicating consistency)
            labels_incons = [1-x for x in labels]
            finegrained_labels_incons = [1-x for x in finegrained_labels]
            labels = labels + labels_incons
            finegrained_labels = finegrained_labels + finegrained_labels_incons
        
        tokenized_sequences = self.tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
        if (self.config['energynet']['output_form'] == 'real_num') and (self.config['energynet']['label_column'] == 'finegrained_labels'):
            labels = torch.Tensor(labels).reshape(-1, 1)
        elif (self.config['energynet']['output_form'] == '2dim_vec') and (self.config['energynet']['label_column'] == 'finegrained_labels'):
            labels = torch.Tensor(labels).reshape(-1, 1)
            labels = torch.tile(labels, (1,2))
            labels[:, 0] = 1 - labels[:, 1] 
        elif (self.config['energynet']['label_column'] == 'binary_labels') or (self.config['energynet']['label_column'] == 'original_labels'):
            labels = torch.LongTensor(labels)
        elif (self.config['energynet']['label_column'] == '3class_finegrained_labels'):
            labels = torch.Tensor(labels).reshape(-1, 3)
        
        return {'input_ids': tokenized_sequences['input_ids'].to(self.config['device']), 
                'attention_mask': tokenized_sequences['attention_mask'].to(self.config['device']), 
                'labels': labels.to(self.config['device']),
                'finegrained_labels': torch.Tensor(finegrained_labels).to(self.config['device'])}

    def get_dataloader(self, dataset=None, batch_size=None, batch_sampler=None, shuffle=False):
        if batch_sampler:
            return DataLoader(dataset, batch_sampler = batch_sampler, collate_fn=self.collate_fn)
        else:
            return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=self.collate_fn)
        