from .ebm import EBM
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class CausalLMEnergyModel(EBM):
    """Causal LM-based energy model for measuring sequence fluency"""

    def __init__(self, model_path: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16, args:dict = {}):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, dtype=dtype)
        self.model.eval()
        self.args = args
    
    @property
    def device(self):
        """Get device from model parameters"""
        return next(self.model.parameters()).device

    def calculate_energy(self, prefix: str, generations: List[str]):
        """ Calculate negative log likelihood of the generations considering the prefix. If prefix does not exist, pass "" or " " as the prefix."""
        if (prefix == "") or (prefix == " "):
            prefix = " "
        
        if self.args.get("apply_chat_template", False):
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prefix},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = prefix
        
        prompt_enc = self.tokenizer(prompt_text, 
                                    add_special_tokens=False, 
                                    return_tensors="pt").to(self.device)

        generations_enc = self.tokenizer(generations, 
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                        padding=True).to(self.device)

        input_tokens = torch.cat([prompt_enc["input_ids"].expand(len(generations), -1), generations_enc["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_enc["attention_mask"].expand(len(generations), -1), generations_enc["attention_mask"]], dim=1)
        
        with torch.no_grad():
            model_output = self.model(input_ids=input_tokens, attention_mask=attention_mask)
            
        lm_logits = model_output[0][:, prompt_enc["input_ids"].size(1)-1:-1, :]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        # input dimensions : (N, V, L), (N, L)
        energy = F.nll_loss(lm_logprobs.permute(0, 2, 1), generations_enc["input_ids"], reduction="none")
        energy = energy * generations_enc["attention_mask"]
        
        energy = energy.sum(dim=-1)
        
        if self.args and self.args.get("length_normalize", False):
            total_tokens = generations_enc["attention_mask"].sum(dim=-1).clamp(min=1)
            energy /= torch.pow(total_tokens, self.args.get("length_normalize_power", 1.0))
        
        return energy
        