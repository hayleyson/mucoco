import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class EncoderModel(nn.Module):

    def __init__(self, params):
        super(EncoderModel, self).__init__()
        if params['locate']['type'] == 'gradnorm':
            self.base_model = AutoModel.from_pretrained(params['energynet']['base_model'], output_hidden_states=True)
        else:
            self.base_model = AutoModel.from_pretrained(params['energynet']['base_model'])
        self.tokenizer = AutoTokenizer.from_pretrained(params['energynet']['base_model'])
        # self.ReLU = nn.ReLU()
        self.hidden_dim = self.base_model.config.hidden_size
        self.params = params
        self.output_form = self.params['energynet']['output_form']
        self.linear1 = None
        self.initialize()
        
        # special tokens
        # single sequence: <s> X </s>
        # pair of sequences: <s> A </s></s> B </s>
        
    def forward(self, input_ids, attention_mask):
        """forward function

        Args:
            inputs (tuple): two elements, 'input_ids', 'attention_mask'
            'input_ids': LongTensor. shape=(bat_size, seq_len)
            'mask': LongTensor. shape=(bat_size, seq_len)

        Returns:
            output: prediction of relationship . shape=(bat_size, 1)
        """
        output, hidden_states = None, None
            
        if self.params['locate']['type'] == 'gradnorm': 
            
            output_all = self.base_model(input_ids = input_ids,
                            attention_mask = attention_mask)
            output = output_all[0][:,0,:] ## taking CLS token representation
            output = output.squeeze(1) ## squeeze out sequence length dimension
            output = self.linear1(output)
            hidden_states = output_all['hidden_states'] ## return hidden states of embedding layer
            
        else:
            
            with torch.no_grad():
                output = self.base_model(input_ids = input_ids,
                                attention_mask = attention_mask)[0][:,0,:] ## taking CLS token representation
                output = output.squeeze(1) ## squeeze out sequence length dimension
                output = self.linear1(output)
        
        return output, hidden_states
    
    def initialize(self):
            
        # depending on the output dimension, define the corresponding final layer.
        if self.output_form == 'real_num':
            self.linear1 = nn.Linear(self.hidden_dim, 1) # Regard output as a compatibility score (a single real value)
        elif self.output_form == '2dim_vec':
            self.linear1 = nn.Linear(self.hidden_dim, 2) # Regard output as a classification result = (consistent, in_consistent)
        elif self.output_form == '3dim_vec':
            self.linear1 = nn.Linear(self.hidden_dim, 3)
        else:
            raise