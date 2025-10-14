"""
This code is adapted from Mucola's losses module. (https://github.com/Sachin19/mucoco/blob/sampling2/mucoco/losses)
"""
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_module.losses import BaseLoss, register_loss

@register_loss("sc_energy")
class SCEnergy(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = args.device

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id    

    def compute_gold_loss(self, prompt:str, prediction:List[str], **kwargs):
        """
        c.f. prompt column is not used. 
        """
        
        # set consistency verification
        output, _ = self.model.energy_model(prediction, pair_only = True)
        # print(f"output: {output}")
        
        if (self.model.output_form == 'real_num'):
            return output.reshape(-1)
        elif (self.model.output_form == '2dim_vec'):
            return output[:,1].reshape(-1) # label_id hardcoded to 1
        else:
            raise NotImplementedError
        