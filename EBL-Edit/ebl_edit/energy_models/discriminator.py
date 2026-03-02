from .ebm import EBM
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DiscriminatorEnergyModel(EBM):
    
    def __init__(self, model_path: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16, args:dict = {}):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device_map, dtype=dtype)
        self.model.eval()
        self.args = args
    
    @property
    def device(self):
        """Get device from model parameters"""
        return next(self.model.parameters()).device

    def calculate_energy(self, prefix: str, generations: List[str]):
        """Calculate negative log probability of target class given the prefix and generations"""
        
        if (prefix == "") or (prefix == " "):
            if self.args.get("task", None) == "nli":
                raise ValueError("Premise must be provided for NLI task.")
            sequences = generations
        else:
            if self.args.get("task", None) == "nli":
                sequences = [self.tokenizer.bos_token + prefix + self.tokenizer.sep_token + seq + self.tokenizer.eos_token for seq in generations]
            else:
                sequences = ["".join([prefix, seq]) if seq.startswith(" ") else " ".join([prefix, seq]) for seq in generations]
                
        tokenized_sequences = self.tokenizer(sequences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**tokenized_sequences)
        
        logits = model_output[0]
        logprobs = F.log_softmax(logits, dim=-1)
        loss = -logprobs[:, self.args.get("label_id", None)]
        
        return loss