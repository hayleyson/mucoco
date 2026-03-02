import joblib
import json
import argparse
import os
import time
from typing import List

import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import baseLM

class HuggingfaceLM(baseLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16)



class HFVanillaLM(HuggingfaceLM):
    def generate(self, prefix: str, n: int, max_new_tokens: int, top_p: float = 0.96) -> List[str]:
        
        prefix_encoded = self.tokenizer(prefix, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**prefix_encoded, 
                                        max_new_tokens=max_new_tokens,
                                        num_return_sequences=n,
                                        do_sample=True,
                                        top_p=top_p, 
                                        temperature=1.0)
        
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generations = [x[len(prefix):] for x in generated_texts]
        
        return generations


class HFChatLM(HuggingfaceLM):

    def __init__(self, model_name: str, task: str, prompt_type: str):
        super().__init__(model_name)
        self.set_prompt(task, prompt_type)

    def generate(self, prefix: str, n: int, max_new_tokens: int, top_p: float = 0.96) -> List[str]:
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            # If the prompt includes {prefix}, inject it. Otherwise use the prompt as is
            {"role": "user", "content": self.user_prompt.format(prefix=prefix) if "{prefix}" in self.user_prompt else self.user_prompt}
        ]
        
        prefix_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_encoded = self.tokenizer(prefix_text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**prefix_encoded, 
                                        max_new_tokens=max_new_tokens,
                                        num_return_sequences=n,
                                        do_sample=True,
                                        top_p=top_p, 
                                        temperature=1.0)
        
        generated_ids = generated_ids[:, prefix_encoded.input_ids.shape[-1]:]
        generations = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generations