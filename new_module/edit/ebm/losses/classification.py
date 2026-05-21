"""
This code is adapted from Mucola's losses module. (https://github.com/Sachin19/mucoco/blob/sampling2/mucoco/losses)
"""
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_module.edit.ebm.losses import BaseLoss, register_loss

@register_loss("classification")
class Classification(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id    

    def compute_gold_loss(self, prompt:str, prediction:List[str], label_id, **kwargs):
        '''
        compute the loss wrt concatenated prompt and prediction
        '''
        if self.args.task == "nli":
            sequences = [self.tokenizer.bos_token + prompt + self.tokenizer.sep_token + h + self.tokenizer.eos_token for h in prediction]
        else:
            sequences = [prompt + " " + h for h in prediction]
            
        tokenized_sequences = self.tokenizer(sequences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        
        model_output = self.model(**tokenized_sequences)
        lm_logits = model_output[0]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)
        loss = -lm_logprobs[:, label_id]
        return loss
