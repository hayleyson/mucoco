import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model


"""
Name                        GPU     vector-dim  max-token
=============================================================
gte-large-en-v1.5(335M)         : 1.25GB,   1024 dim,   8192   token
gte-Qwen1.5-7B-instruct : 26.45GB,  4096 dim,   32000 token
gte-Qwen2-1.5B-instruct : 6.62GB,   1536 dim    1536  token
gte-Qwen2-7B-instruct   : 26.45GB,  3584 dim    32000 token
===============================================================

code and info source : https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct


[sh file script example]
    srun python train_wdb_Set_Contrastive.py \
        --task vqa \
        --dataset lconvqa \
        --loss_type triplet \
        --repre_model gte-Qwen2-7B-instruct

"""

from_modelname_to_vectordim = {
    "gte-large-en-v1.5":                1024,
    "gte-Qwen1.5-7B-instruct":  4096,
    "gte-Qwen2-1.5B-instruct":  1536,
    "gte-Qwen2-7B-instruct":    3584,
}

class gte_qwen(nn.Module):

    def __init__(self, params):
        super(gte_qwen, self).__init__()

        self.model = AutoModel.from_pretrained(f"Alibaba-NLP/{params['energynet']['repre_model']}", trust_remote_code=True)
        # print("model:")
        # print(self.model)
        # peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
        # self.model = get_peft_model(self.model, peft_config)
        # print(self.model.print_trainable_parameters())
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"Alibaba-NLP/{params['energynet']['repre_model']}", trust_remote_code=True)
        self.ReLU = nn.ReLU()
        self.params = params
        self.output_form = self.params['energynet']['output_form']
        self.linear1 = None
        self.sigmoid = nn.Sigmoid()
        self.initialize()
        
        
        
    def forward(self, inputs):
        """forward function

        Args:
            inputs (tuple): two elements, 'input_ids', 'attention_mask'
            'input_ids': LongTensor. shape=(bat_size, seq_len)
            'mask': LongTensor. shape=(bat_size, seq_len)

        Returns:
            output: prediction of relationship . shape=(bat_size, 1)
        """
        # print("inputs")
        # print(type(inputs), len(inputs))
        if len(inputs)==2:
            input_tensor, mask = inputs['input_ids'], inputs['attention_mask']
            input_tensor, mask = torch.tensor(input_tensor).to(self.params['device']), torch.tensor(mask).to(self.params['device'])
        else:
            input_tensor = inputs
            mask = None
    
        if mask == None:
            mask = torch.ones_like(input_tensor)
            
        # print(f"input_tensor.shape:{input_tensor.shape}")
        # print(f"mask.shape:{mask.shape}")
        output_all = self.model(input_ids = input_tensor,
                        attention_mask = mask) # keys: ['last_hidden_state', 'past_key_values']
        # forward() got an unexpected keyword argument 'prompt_name'
        embeddings = last_token_pool(output_all.last_hidden_state, mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # print(f"embeddings.shape:{embeddings.shape}")
        output = output_all[0][:,0,:]
        # print("output_all shape:", output_all[0].shape)
        output = torch.reshape(output, (-1, from_modelname_to_vectordim[self.params['energynet']['repre_model']]))
        norms = torch.norm(output, dim = -1).detach().unsqueeze(-1)
        output = output / norms
        output = self.linear1(output)
        
        if self.output_form == 'real_num':
            if self.params['locate']['type'] == 'gradnorm': 
                hidden_states = output_all['hidden_states'][0] ## return hidden states of embedding layer
                return self.sigmoid(output), hidden_states
                
            else:
                return self.sigmoid(output)
        elif self.output_form == '2dim_vec':
            return output
        
    def initialize(self):

        # depending on the output dimension, define the corresponding final layer.
        if self.output_form == 'real_num':
            self.linear1 = nn.Linear(from_modelname_to_vectordim[self.params['energynet']['repre_model']], 1) # Regard output as a compatibility score (a single real value)
        elif self.output_form == '2dim_vec':
            self.linear1 = nn.Linear(from_modelname_to_vectordim[self.params['energynet']['repre_model']], 2) # Regard output as a classification result = (consistent, in_consistent)
        


def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# # Each query must come with a one-sentence instruction that describes the task
# task = 'Given a web search query, retrieve relevant passages that answer the query'
# queries = [
#     get_detailed_instruct(task, 'how much protein should a female eat'),
#     get_detailed_instruct(task, 'summit define')
# ]
# # No need to add instruction for retrieval documents
# documents = [
#     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
# ]
# input_texts = queries + documents

# max_length = 8192

# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
# outputs = model(**batch_dict)
# embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# # normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:2] @ embeddings[2:].T) * 100
# print(scores.tolist())
