import torch, numpy, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
import transformers

def seed_worker(worker_id=0):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(1014)

class no_decomposition(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.representation_model = None
        self.tokenizer = None
        
    def detect_span(self, set_text, sep_token: str = '.', cls_token: str = '<s>'):
        
        """
        Hardcoded seperator token which is '.' and cls token which is '<s>'
        """

        # set_text == text, e.g., '<s> qa pair 1 </s> qa pair 2 ... </s>

        out = set_text[len(cls_token):].split(sep_token)[:-1]
        
        return [o+sep_token for o in out]
        
    def instance_preserving_encode_plus(self, string_inputs: List[str]) -> torch.Tensor:
        """
        By default, padding = True, truncation = True, add_special_tokens = False
        """
        
        
        # Detect spans for each input string
        instances_per_batch = [self.detect_span(string) for string in string_inputs]
        
        # Encode each instance
        instance_encoded_per_batch = []        
        for instances in instances_per_batch:
            instance_encoded = [
                self.tokenizer.encode(instance, add_special_tokens=False)
                for instance in instances
            ]
            # flatten the list
            instance_encoded = sum(instance_encoded, [])
            instance_encoded = [self.tokenizer.cls_token_id] + instance_encoded
            instance_encoded_per_batch.append(torch.LongTensor(instance_encoded))
            
        input_tensor = torch.nn.utils.rnn.pad_sequence(instance_encoded_per_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_tensor = input_tensor[:, :self.tokenizer.model_max_length]
        mask = (input_tensor != self.tokenizer.pad_token_id).long()
        
        return transformers.BatchEncoding({"input_ids": input_tensor, 
                                           "attention_mask": mask},
                                          tensor_type="pt").to(self.params['device'])
        
    def forward(self, pair, pair_only = False):

        # print("pair:")
        # print(pair)
        # print("-----")
        if pair_only:
            inputs = pair
        else:
            inputs = [p[0] for p in pair]
        
        # print("inputs:", inputs)
        inputs = self.instance_preserving_encode_plus(inputs)
        outputs = self.representation_model(inputs)
        return outputs

    def set_representation_model(self, representation_model):
        self.representation_model = representation_model
        self.tokenizer = self.representation_model.tokenizer


class no_decomposition_loader():
    def __init__(self, dataset = None,  tokenizer = None, params = None):
        super().__init__()
        self.dataset_text = dataset.dataset
        self.params = params

        # data_name attribute is used for mixed-domain training
        if not hasattr(dataset, 'data_name'):
            self.data_name = None
        else:
            self.data_name = dataset.data_name
        if self.data_name == None:
            self.data_name = [self.params['dataset'] for _ in range(len(self.dataset_text))]
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.dataset_decompose = []        

    def decompose_transform(self):
        for idx, data in enumerate(self.dataset_text):
            tmp = f"{self.cls_token} "
            consistency_list = []
            if self.data_name[idx] in {'lconvqa'}:
                for i, xy_pair in enumerate(data):
                    x = xy_pair[0]
                    y = xy_pair[1]
                    if xy_pair[2] == False:
                        consistency_list.append(i)
                    if i!=0:
                        tmp += " "
                    tmp += x
                    tmp += f" The answer is "
                    tmp += y
                    if all([not tmp.endswith(mark) for mark in ['.', '!', '?']]):
                        tmp += "."
                self.dataset_decompose.append([tmp, consistency_list, data])
            
            elif self.data_name[idx] in {'set_nli'}:
                for i, xy_pair in enumerate(data):
                    x = xy_pair[0]
                    if xy_pair[1] == False:
                        consistency_list.append(i)
                    if i!=0:
                        tmp += " "
                    tmp += x
                    if all([not tmp.endswith(mark) for mark in ['.', '!', '?']]):
                        tmp += "."
                    if tmp.endswith(".."):
                        tmp = tmp[:-1]
                self.dataset_decompose.append([tmp, consistency_list, data])
            else:
                print("no decomposition loader error.")
                print("invalid dataset name. Current is", self.data_name[idx])
        self.dataset_text = None

    def decompose_transform_mto(self):
        for idx, data in enumerate(self.dataset_text):
            tmp = f"{self.cls_token} "
            consistency_list = []
            if self.data_name[idx] in {'lconvqa'}:
                for i, xy_pair in enumerate(data):
                    x = xy_pair[0]
                    y = xy_pair[1]
                    if xy_pair[2] == False:
                        consistency_list.append(i)
                    if i!=0:
                        tmp += " "
                    if i == len(data)-1:
                        tmp += " " + self.sep_token + " "
                    tmp += x
                    tmp += f" The answer is "
                    tmp += y
                    if all([not tmp.endswith(mark) for mark in ['.', '!', '?']]):
                        tmp += "."
                self.dataset_decompose.append([tmp, consistency_list, data])
            
            elif self.data_name[idx] in {'set_nli'}:
                for i, xy_pair in enumerate(data):
                    x = xy_pair[0]
                    if xy_pair[1] == False:
                        consistency_list.append(i)
                    if i!=0:
                        tmp += " "
                    if i == len(data)-1:
                        tmp += " " + self.sep_token + " "
                    tmp += x
                    if all([not tmp.endswith(mark) for mark in ['.', '!', '?']]):
                        tmp += "."
                    if tmp.endswith(".."):
                        tmp = tmp[:-1]
                self.dataset_decompose.append([tmp, consistency_list, data])
            else:
                print("no decomposition loader error.")
                print("invalid dataset name. Current is", self.data_name[idx])
        self.dataset_text = None

    def get_loader(self, split = 'train', MtO_mode = False):
        if not MtO_mode:
            self.decompose_transform()
        else:
            self.decompose_transform_mto()
        if split == 'train':
            return DataLoader(self.dataset_decompose, self.params['batch_size'], shuffle = False, collate_fn = collate_fn, generator = g)
        else:
            return DataLoader(self.dataset_decompose, self.params['eval']['batch_size'], shuffle = False, collate_fn = collate_fn, generator = g)



def collate_fn(batch):
    return batch
