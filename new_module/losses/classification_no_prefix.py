import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_module.losses import BaseLoss, register_loss

@register_loss("classification_no_prefix_logprobloss")
class ClassificationLogProbLoss(BaseLoss):

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
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        if self.args.task == "nli":
            premises, hypotheses = list(map(list, zip(*prediction)))
            prediction = self.tokenizer.batch_encode_plus(premises, hypotheses, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True).to(self.device)
        else:
            prediction = [x + self.tokenizer.eos_token + self.tokenizer.eos_token for x in prediction]
            prediction = self.tokenizer.batch_encode_plus(prediction, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True).to(self.device)
        model_output = self.model(**prediction)
        lm_logits = model_output[0]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)
        loss = -lm_logprobs[:, label_id]
        return loss
