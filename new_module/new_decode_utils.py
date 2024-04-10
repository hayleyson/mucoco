import logging
import os
from typing import List, Tuple
from itertools import product
import math

import torch
import torch.nn.functional as F
import transformers
import wandb
from torch.utils.data import DataLoader,Dataset

import new_module.losses as lossbuilder

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'DEBUG').upper(), 
                    format='%(message)s')
logger = logging.getLogger(__name__)

# def get_beam_hypotheses(source_text:str, 
#                     masked_sequence:torch.Tensor, 
#                     indices_in_mlm_tokens:Tuple[torch.Tensor],
#                     predicted_token_ids:torch.Tensor,
#                     mlm_tokenizer:transformers.AutoTokenizer, 
#                     lossfns:List[lossbuilder.BaseLoss],
#                     config:dict) -> List[List[str]]:
#     """
#     A function to get hypotheses of beam size via editing beam search with reranking.
#     Run this function if config['method'] == 'mlm-beamsearch-v0' or config['method'] == 'mlm-beamsearch-v1'
#     If config['method'] == 'mlm-beamsearch-v1', rerank beam only with fluency energy.
#     If config['method'] == 'mlm-beamsearch-v0', rerank beam with a weighted sum of fluency and constraint energy.
    
#     #ToDo
#     #Implement mlm-beamsearch-v0 with allsat-primary and compare 
    
#     params: 
#         source_text: a prompt text 
#         masked_sequence: token ids of original generation text with located indices masked. tokenized by MLM's tokenizer.
#         indices_in_mlm_tokens: a result of running 
#                                     `indices_in_mlm_tokens = (
#                                                                 inputs.input_ids == mlm_tokenizer.mask_token_id
#                                                                 ).nonzero(as_tuple=True)`
#         predicted_token_ids: a result of running
#                                     `predicted_token_ids = torch.topk(
#                                                                 logits[indices_in_mlm_tokens[0], indices_in_mlm_tokens[1], :],
#                                                                 k=config['k_per_location'],
#                                                                 dim=-1,).indices`
#         mlm_tokenizer: tokenizer of MLM
#         lossfns: a list of loss functions
#         config: a dictionary of configurations
    
#     returns:
#         hypotheses: a list of a list of the beam number of hypotheses for each sample         
#     """
    
#     def repeat_interleave_unravel(arr,split_blocks):
#         arr_ = torch.split(arr.T,1,dim=1)
#         arr_ = [x.repeat(1,split_blocks[i]).reshape(-1,1) for i,x in enumerate(arr_)]
#         arr_ = torch.cat(arr_,dim=0)
#         return arr_
    
#     hypotheses = list(torch.split(masked_sequence,1,dim=0)) ## [torch.tensor([[a],[b],[c]]), torch.tensor([[d]])]
#     edit_indices = sorted(list(set(indices_in_mlm_tokens[1].tolist())))
#     for curr_edit_index in edit_indices:
        
#         batch_ids_to_edit = indices_in_mlm_tokens[0][indices_in_mlm_tokens[1]==curr_edit_index].tolist()
#         num_initial_hypotheses = [len(hypotheses[i]) for i in batch_ids_to_edit] ## keep track of initial hypotheses count e.g. [3, 1]
#         tmp_hypotheses = [hypotheses[i].repeat((config['k_per_location'],1)) for i in batch_ids_to_edit] ## [torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c]]), torch.tensor([[d],[d],[d]])]
#         num_initial_tmp_hypotheses = [len(x) for x in tmp_hypotheses]
#         tmp_hypotheses = torch.cat(tmp_hypotheses,dim=0) ## torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c],[d],[d],[d]])
        

#         new_func_candidates = predicted_token_ids[indices_in_mlm_tokens[1]==curr_edit_index] ## shape: (len(batch_ids_to_edit), k_per_location) e.g. [[x,y,z],[q,w,e]]
#         new_func_candidates = repeat_interleave_unravel(new_func_candidates,num_initial_hypotheses) ## shape: (sum(num_initial_hypotheses), k_per_location) e.g. [[x],[x],[x],[y],[y],[y],[z],[z],[z],[q],[w],[e]]
#         new_func_candidates = new_func_candidates.to(config['device'])
        

#         tmp_hypotheses = torch.cat((tmp_hypotheses[ :, :curr_edit_index], new_func_candidates),dim=-1) ## tmp_hypotheses: [(a,b,c),(a,b,c), ..., (a,b,c)], new_func_candidates: [(p,p,p), (q,q,q), ..., (v,v,v)]

#         loss_weights = [1 - config['closs_weight'], config['closs_weight']]
#         curr_loss = torch.zeros(tmp_hypotheses.shape[0]).to(config['device'])
#         for lossid, lossname in enumerate(config["losses"]):
#             if config['method'] == 'mlm-beamsearch-v1' and lossid > 0:
#                 continue
#             with torch.no_grad():
#                 lossvalue = lossfns[lossid].compute_gold_loss(
#                     source_text, mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True),
#                     label_id=config['target_label_ids'][lossid],
#                 )
#             torch.cuda.empty_cache()
#             curr_loss += loss_weights[lossid] * lossvalue
#         curr_loss = torch.split(curr_loss, num_initial_tmp_hypotheses, dim=0)
#         top_beams = [torch.topk(x, k=config['beam_size'], dim=-1, largest=False).indices for x in curr_loss]

#         tmp_hypotheses = torch.split(tmp_hypotheses, num_initial_tmp_hypotheses, dim=0)
#         for jx, ix in enumerate(batch_ids_to_edit):

#             hypotheses[ix] = torch.cat([tmp_hypotheses[jx][top_beams[jx]], masked_sequence[ix][curr_edit_index+1:].unsqueeze(0).repeat(config['beam_size'],1)], dim=-1)
            
#     return [mlm_tokenizer.batch_decode(x, skip_special_tokens=True) for x in hypotheses]

def get_beam_hypotheses_v0(source_text:str, 
                    masked_sequence:torch.Tensor, 
                    indices_in_mlm_tokens:Tuple[torch.Tensor],
                    predicted_token_ids:torch.Tensor,
                    mlm_tokenizer:transformers.AutoTokenizer, 
                    lossfns:List[lossbuilder.BaseLoss],
                    config:dict) -> List[List[str]]:
    """
    A function to get hypotheses of beam size via editing beam search with reranking.
    Run this function if config['method'] == 'mlm-beamsearch-v0'
    Almost the same as get_beam_hypotheses_v1 except the scoring function during beam search.
    If config['method'] == 'mlm-beamsearch-v0', rerank beam with a weighted sum of fluency and constraint energy.
    
    #ToDo
    #Implement mlm-beamsearch-v0 with allsat-primary and compare 
    
    params: 
        source_text: a prompt text 
        masked_sequence: token ids of original generation text with located indices masked. tokenized by MLM's tokenizer.
        indices_in_mlm_tokens: a result of running 
                                    `indices_in_mlm_tokens = (
                                                                inputs.input_ids == mlm_tokenizer.mask_token_id
                                                                ).nonzero(as_tuple=True)`
        predicted_token_ids: a result of running
                                    `predicted_token_ids = torch.topk(
                                                                logits[indices_in_mlm_tokens[0], indices_in_mlm_tokens[1], :],
                                                                k=config['k_per_location'],
                                                                dim=-1,).indices`
        mlm_tokenizer: tokenizer of MLM
        lossfns: a list of loss functions
        config: a dictionary of configurations
    
    returns:
        hypotheses: a list of a list of the beam number of hypotheses for each sample         
    """
    
    def repeat_interleave_unravel(arr,split_blocks):
        arr_ = torch.split(arr.T,1,dim=1)
        arr_ = [x.repeat(1,split_blocks[i]).reshape(-1,1) for i,x in enumerate(arr_)]
        arr_ = torch.cat(arr_,dim=0)
        return arr_
    
    hypotheses = list(torch.split(masked_sequence,1,dim=0)) ## [torch.tensor([[a],[b],[c]]), torch.tensor([[d]])]
    edit_indices = sorted(list(set(indices_in_mlm_tokens[1].tolist())))
    for curr_edit_index in edit_indices:
        
        batch_ids_to_edit = indices_in_mlm_tokens[0][indices_in_mlm_tokens[1]==curr_edit_index].tolist()
        num_initial_hypotheses = [len(hypotheses[i]) for i in batch_ids_to_edit] ## keep track of initial hypotheses count e.g. [3, 1]
        tmp_hypotheses = [hypotheses[i].repeat((config['k_per_location'],1)) for i in batch_ids_to_edit] ## [torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c]]), torch.tensor([[d],[d],[d]])]
        num_initial_tmp_hypotheses = [len(x) for x in tmp_hypotheses]
        tmp_hypotheses = torch.cat(tmp_hypotheses,dim=0) ## torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c],[d],[d],[d]])
        

        new_func_candidates = predicted_token_ids[indices_in_mlm_tokens[1]==curr_edit_index] ## shape: (len(batch_ids_to_edit), k_per_location) e.g. [[x,y,z],[q,w,e]]
        new_func_candidates = repeat_interleave_unravel(new_func_candidates,num_initial_hypotheses) ## shape: (sum(num_initial_hypotheses), k_per_location) e.g. [[x],[x],[x],[y],[y],[y],[z],[z],[z],[q],[w],[e]]
        new_func_candidates = new_func_candidates.to(config['device'])
        

        tmp_hypotheses = torch.cat((tmp_hypotheses[ :, :curr_edit_index], new_func_candidates),dim=-1) ## tmp_hypotheses: [(a,b,c),(a,b,c), ..., (a,b,c)], new_func_candidates: [(p,p,p), (q,q,q), ..., (v,v,v)]

        loss_weights = [1 - config['closs_weight'], config['closs_weight']]
        curr_loss = torch.zeros(tmp_hypotheses.shape[0]).to(config['device'])
        for lossid, lossname in enumerate(config["losses"]):
            with torch.no_grad():
                lossvalue = lossfns[lossid].compute_gold_loss(
                    source_text, mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True),
                    label_id=config['target_label_ids'][lossid],
                )
            torch.cuda.empty_cache()
            curr_loss += loss_weights[lossid] * lossvalue
        curr_loss = torch.split(curr_loss, num_initial_tmp_hypotheses, dim=0)
        top_beams = [torch.topk(x, k=config['beam_size'], dim=-1, largest=False).indices for x in curr_loss]

        tmp_hypotheses = torch.split(tmp_hypotheses, num_initial_tmp_hypotheses, dim=0)
        for jx, ix in enumerate(batch_ids_to_edit):

            hypotheses[ix] = torch.cat([tmp_hypotheses[jx][top_beams[jx]], masked_sequence[ix][curr_edit_index+1:].unsqueeze(0).repeat(config['beam_size'],1)], dim=-1)
            
    return [mlm_tokenizer.batch_decode(x, skip_special_tokens=True) for x in hypotheses]

def get_beam_hypotheses_v1(source_text:str, 
                    masked_sequence:torch.Tensor, 
                    indices_in_mlm_tokens:Tuple[torch.Tensor],
                    predicted_token_ids:torch.Tensor,
                    mlm_tokenizer:transformers.AutoTokenizer, 
                    lossfns:List[lossbuilder.BaseLoss],
                    config:dict) -> List[List[str]]:
    """
    A function to get hypotheses of beam size via editing beam search with reranking.
    Run this function if config['method'] == 'mlm-beamsearch-v1'
    Almost the same as get_beam_hypotheses_v0 except the scoring function during beam search.
    If config['method'] == 'mlm-beamsearch-v1', rerank beam only with fluency energy.
    If config['method'] == 'mlm-beamsearch-v0', rerank beam with a weighted sum of fluency and constraint energy.
    
    params: 
        source_text: a prompt text 
        masked_sequence: token ids of original generation text with located indices masked. tokenized by MLM's tokenizer.
        indices_in_mlm_tokens: a result of running 
                                    `indices_in_mlm_tokens = (
                                                                inputs.input_ids == mlm_tokenizer.mask_token_id
                                                                ).nonzero(as_tuple=True)`
        predicted_token_ids: a result of running
                                    `predicted_token_ids = torch.topk(
                                                                logits[indices_in_mlm_tokens[0], indices_in_mlm_tokens[1], :],
                                                                k=config['k_per_location'],
                                                                dim=-1,).indices`
        mlm_tokenizer: tokenizer of MLM
        lossfns: a list of loss functions
        config: a dictionary of configurations
    
    returns:
        hypotheses: a list of a list of the beam number of hypotheses for each sample         
    """
    
    def repeat_interleave_unravel(arr,split_blocks):
        arr_ = torch.split(arr.T,1,dim=1)
        arr_ = [x.repeat(1,split_blocks[i]).reshape(-1,1) for i,x in enumerate(arr_)]
        arr_ = torch.cat(arr_,dim=0)
        return arr_
    
    hypotheses = list(torch.split(masked_sequence,1,dim=0)) ## [torch.tensor([[a],[b],[c]]), torch.tensor([[d]])]
    edit_indices = sorted(list(set(indices_in_mlm_tokens[1].tolist())))
    for curr_edit_index in edit_indices:
        
        batch_ids_to_edit = indices_in_mlm_tokens[0][indices_in_mlm_tokens[1]==curr_edit_index].tolist()
        num_initial_hypotheses = [len(hypotheses[i]) for i in batch_ids_to_edit] ## keep track of initial hypotheses count e.g. [3, 1]
        tmp_hypotheses = [hypotheses[i].repeat((config['k_per_location'],1)) for i in batch_ids_to_edit] ## [torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c]]), torch.tensor([[d],[d],[d]])]
        num_initial_tmp_hypotheses = [len(x) for x in tmp_hypotheses]
        tmp_hypotheses = torch.cat(tmp_hypotheses,dim=0) ## torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c],[d],[d],[d]])
        

        new_func_candidates = predicted_token_ids[indices_in_mlm_tokens[1]==curr_edit_index] ## shape: (len(batch_ids_to_edit), k_per_location) e.g. [[x,y,z],[q,w,e]]
        new_func_candidates = repeat_interleave_unravel(new_func_candidates,num_initial_hypotheses) ## shape: (sum(num_initial_hypotheses), k_per_location) e.g. [[x],[x],[x],[y],[y],[y],[z],[z],[z],[q],[w],[e]]
        new_func_candidates = new_func_candidates.to(config['device'])
        

        tmp_hypotheses = torch.cat((tmp_hypotheses[ :, :curr_edit_index], new_func_candidates),dim=-1) ## tmp_hypotheses: [(a,b,c),(a,b,c), ..., (a,b,c)], new_func_candidates: [(p,p,p), (q,q,q), ..., (v,v,v)]

        with torch.no_grad():
            lossvalue = lossfns[0].compute_gold_loss(
                source_text, mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True),
                label_id=config['target_label_ids'][0],
            )
        torch.cuda.empty_cache()
        
        curr_loss = torch.split(lossvalue, num_initial_tmp_hypotheses, dim=0)
        top_beams = [torch.topk(x, k=config['beam_size'], dim=-1, largest=False).indices for x in curr_loss]

        tmp_hypotheses = torch.split(tmp_hypotheses, num_initial_tmp_hypotheses, dim=0)
        for jx, ix in enumerate(batch_ids_to_edit):

            hypotheses[ix] = torch.cat([tmp_hypotheses[jx][top_beams[jx]], masked_sequence[ix][curr_edit_index+1:].unsqueeze(0).repeat(config['beam_size'],1)], dim=-1)
            
    return [mlm_tokenizer.batch_decode(x, skip_special_tokens=True) for x in hypotheses]

def get_combi_hypotheses(masked_sequence:torch.Tensor, 
                 indices_in_mlm_tokens:tuple,
                 predicted_token_ids:torch.Tensor,
                 mlm_tokenizer:transformers.AutoTokenizer,
                 config:dict) -> List[List[str]]:
    """
    A function to get hypotheses of k**l size via getting combinations of candidates per location.
    Run this function if config['method'] == 'mlm-reranking'.
    
    params: 
        masked_sequence: token ids of original generation text with located indices masked. tokenized by MLM's tokenizer.
        indices_in_mlm_tokens: a result of running 
                                    `indices_in_mlm_tokens = (
                                                                inputs.input_ids == mlm_tokenizer.mask_token_id
                                                                ).nonzero(as_tuple=True)`
        predicted_token_ids: a result of running
                                    `predicted_token_ids = torch.topk(
                                                                logits[indices_in_mlm_tokens[0], indices_in_mlm_tokens[1], :],
                                                                k=config['k_per_location'],
                                                                dim=-1,).indices`
        mlm_tokenizer: tokenizer of MLM
        config: a dictionary of configurations
    
    returns:
        hypotheses: a list of a list of k**l number of hypotheses for each sample         
    """

    k = config['k_per_location']
    hypotheses = []
    num_batches = masked_sequence.shape[0]
    for i in range(num_batches):
        
        l = (indices_in_mlm_tokens[0] == i).sum().item()
        tok_cand_combos = list(product(range(k),repeat=l))
        
        tmp_hypotheses = masked_sequence[i,:].repeat((k**l,1))
        tmp_hypotheses[:, indices_in_mlm_tokens[1][indices_in_mlm_tokens[0] == i]] = \
            predicted_token_ids[indices_in_mlm_tokens[0] == i, tok_cand_combos]
            
        tmp_dec_seq = mlm_tokenizer.batch_decode(
                    tmp_hypotheses, skip_special_tokens=True
            )
        hypotheses.append(tmp_dec_seq)
    return hypotheses


def final_reranking(source_text:str,
                    hypotheses:List[List[str]],
                    lossfns:List[lossbuilder.BaseLoss],
                    config:dict,
                    batch_size:int=64) -> Tuple[List[str],torch.FloatTensor,torch.BoolTensor,torch.FloatTensor]:
    """
    
    params: 
        source_text: a prompt text 
        hypotheses: a list of [a list of hypotheses] for each sample       
        lossfns:
        config:
        batch_size:             
    
    returns:
        hypotheses: list of one best hypothesis(editing result) for each of original texts. length same as masked_sequence.shape[0]
        best_weighted_loss: torch.FloatTensor of weighted loss for the best hypotheses.
        best_allsat: torch.ByteTensor of indicator(1,0) whether the best hypotheses satisfy cutoff (min_epsilons) for constraint energy score.
        best_logging_loss: torch.FloatTensor of shape (num samples, 2) of fluency energy score and constraint energy score for each best hypothesis.
    """
    
    
    class CustomDataset(Dataset):
        def __init__(self, hypotheses_data:List[str]):
            self.hypotheses_data = hypotheses_data
            
        def __len__(self):
            return len(self.hypotheses_data)

        def __getitem__(self, idx:int):
            return self.hypotheses_data[idx]
        
        def __getitems__(self, idx:List[int]):
            return [self.hypotheses_data[j] for j in idx]
    
    final_hypotheses = []
    best_weighted_loss = []
    best_allsat = []
    best_logging_loss = []
    
    loss_weights = [1 - config['closs_weight'], config['closs_weight']]
    
    # for i in tqdm(range(len(hypotheses))):
    for i in range(len(hypotheses)):
        curr_loss = torch.zeros(len(hypotheses[i])).to(config['device'])
        logging_loss = torch.zeros((len(hypotheses[i]),2)).to(config['device'])
        data_loader = DataLoader(CustomDataset(hypotheses[i]),batch_size=batch_size)

        for lossid, lossname in enumerate(config["losses"]):
            lossvalues=[]
            with torch.no_grad():
                for batch in data_loader:
                    lossvalue = lossfns[lossid].compute_gold_loss(
                        source_text, batch,
                        label_id=config['target_label_ids'][lossid],
                    )
                    lossvalues.append(lossvalue)
                    torch.cuda.empty_cache()
            lossvalue = torch.cat(lossvalues,dim=0)
            curr_loss += loss_weights[lossid] * lossvalue
            logging_loss[:, lossid] = lossvalue.clone()
            
        allsat_ix = torch.where(logging_loss[:,1]< -math.log(config["min_epsilons"][0]))[0]
        if (len(allsat_ix) > 0) and (config['selection_criteria'] == "allsat_primary"):
        #if (allsat_ix.shape[0] > 0) and (config['selection_criteria'] == "allsat_primary"):
            best_ix = allsat_ix[curr_loss[allsat_ix].argmin()]
        else: ## in case config['selection_criteria'] == "weighted_sum" or allsat is all False
            best_ix = torch.argmin(curr_loss)

        final_hypotheses.append(hypotheses[i][best_ix])
        best_weighted_loss.append(curr_loss[best_ix].item())
        best_allsat.append(1 if best_ix in allsat_ix else 0)
        best_logging_loss.append(logging_loss[best_ix].cpu().tolist())
    
        del curr_loss, logging_loss
        torch.cuda.empty_cache()
    return final_hypotheses, torch.FloatTensor(best_weighted_loss).to(config['device']), \
            torch.BoolTensor(best_allsat).to(config['device']), torch.FloatTensor(best_logging_loss).to(config['device'])

