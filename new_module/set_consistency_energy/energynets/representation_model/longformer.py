import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel,LongformerTokenizer, AutoTokenizer

class longformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096", device_map = "auto")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        # special tokens
            # cls_token : <s>
            # eos, sep token = </s>
            # pad_token = <pad>
            # mask_token = <mask>
        self.CE = nn.CrossEntropyLoss()
        self.linear = nn.Linear(512, 1)
        self.softmax1 = nn.Softmax(dim = 1)
        self.softmax2 = nn.Softmax(dim = 2)

    def forward(self, input_ids, attention_mask = None, global_attention_mask = None):
        
        # Global attention on the first token
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        # forward propagate
        output = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        output = output.pooler_output # size = (bat_size, hidden_size)
        output = torch.reshape(output, (-1, 768))
        norms = torch.norm(output, dim = -1).detach().unsqueeze(-1)
        output = output / norms
        output = self.linear(output) # size = (bat_size, 1)

        return output
