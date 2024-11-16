import logging
import os
from typing import List, Tuple
from itertools import product
import math
from collections import defaultdict

import re
import torch
import torch.nn.functional as F
import transformers
import wandb
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

import new_module.losses as lossbuilder

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'DEBUG').upper(), 
                    format='%(message)s')
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, hypotheses_data:List[str]):
        self.hypotheses_data = hypotheses_data
        
    def __len__(self):
        return len(self.hypotheses_data)

    def __getitem__(self, idx:int):
        return self.hypotheses_data[idx]
    
    def __getitems__(self, idx:List[int]):
        return [self.hypotheses_data[j] for j in idx]

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

def analyze_span_lengths_and_count(text):
    mask_matches = list(re.finditer('<mask>', text))

    mask_info_dict= defaultdict(list)
    prev_mask = None
    span_count = 0
    curr_span_length = 1
    for i, mask in enumerate(mask_matches):
        
        if i == 0:
            mask_info_dict[span_count].append(i)
            
        else:
            if prev_mask.span()[1] == mask.span()[0]:
                mask_info_dict[span_count].append(i)
                curr_span_length += 1
            else:
                span_count += 1
                mask_info_dict[span_count].append(i)
                curr_span_length = 1
        prev_mask = mask

    span_lengths = []
    for span_id, span_len in mask_info_dict.items():
        
        span_lengths.append(len(span_len))
    return mask_info_dict, span_lengths


def editing_with_delete_variable_replace(source_text:str, test_sent:str, test_sent_span_lengths:List[int], 
                                         mlm:AutoModelForMaskedLM, mlm_tokenizer:AutoTokenizer, 
                                         lossfns:List[lossbuilder.BaseLoss], config: dict, batch_size:int=64) -> \
                                             Tuple[List[str],torch.FloatTensor,torch.BoolTensor,torch.FloatTensor]:
    
    """
    
    params: 
        source_text: a prompt text 
        test_sent: a masked text returned by LocateMachine     
        test_sent_span_lengths: a list of span lenghts for each mask span in the test_sent
        mlm:
        mlm_tokenizer:
        lossfns: 
        config:
        batch_size:             
    
    returns:
        hypotheses: list of one best hypothesis(editing result)
        best_weighted_loss: torch.FloatTensor of weighted loss for the best hypothesis.
        best_allsat: torch.ByteTensor of indicator(1,0) whether the best hypothesis satisfy cutoff (min_epsilons) for constraint energy score.
        best_logging_loss: torch.FloatTensor of shape (num samples, 2) of fluency energy score and constraint energy score for each best hypothesis.
    """
    
    # merge masks
    test_sent_merged = re.sub(r"(<mask>)+", "<mask>", test_sent)
    
    # Max number of mask tokens to replace each span
    max_mask_cnt_per_span = [max(x, config['max_tokens_per_span']) for x in test_sent_span_lengths]

    # Get the span information of merged masks in the test sentence
    mask_spans = [x.span() for x in re.finditer('<mask>',test_sent_merged)]

    special_token_ids = mlm_tokenizer.convert_tokens_to_ids(mlm_tokenizer.all_special_tokens)

    queue = []
    queue.append(test_sent_merged[:mask_spans[0][0]])
    for i in range(len(mask_spans)):
        curr_queue_size = len(queue)
        hypotheses=[list(queue)] # deletion case
        # print(f'working on the {i}th span. current queue size: {curr_queue_size}')
        
        for j in range(1, max_mask_cnt_per_span[i] + 1):
        # for j in range(1, 3):
            # print(f"   * appending {j} masks")
            # Set up sentence for MLM inference (note we append variable number mask and then the rest of the sentence where the other spans are masked)
            curr_full_text_hyp = [base_hyp + "<mask>" * j + test_sent_merged[mask_spans[i][1]:] for base_hyp in queue]
            # print(f"-- curr_masked_text: {curr_full_text_hyp}")
            
            # Tokenize & conduct MLM inference
            inputs = mlm_tokenizer(
                curr_full_text_hyp, return_tensors="pt", padding=True, truncation=True
            )
            inputs = inputs.to(config['device']) 
            masked_sequence=inputs['input_ids']
            
            if config['consider_prompt_for_cand_gen']:
            
                prompt_enc=mlm_tokenizer(mlm_tokenizer.bos_token + source_text,add_special_tokens=False, return_tensors="pt", padding=True, truncation=True).to(config['device'])
                prompt_enc['input_ids']=prompt_enc['input_ids'].expand(curr_queue_size,-1)
                prompt_enc['attention_mask']=prompt_enc['attention_mask'].expand(curr_queue_size,-1)
                # print(f"-- shape of source text after being tokenized and converted to input ids: {prompt_enc['input_ids'].shape}")
                
                input_tokens = torch.cat([prompt_enc.input_ids, inputs.input_ids], dim=1).to(config['device'])
                attention_masks = torch.cat([prompt_enc.attention_mask, inputs.attention_mask], dim=1).to(config['device'])
                
                # with torch.no_grad():
                #     logits = mlm(**inputs).logits
                with torch.no_grad():
                    logits = mlm(input_ids = input_tokens, 
                                attention_mask = attention_masks).logits

                # Choose top k among non-special tokens
                # print(f"-- shape of logits before removing source text part: {logits.shape}")
                logits = logits[:, prompt_enc.input_ids.shape[1]:]
                
            else:
                with torch.no_grad():
                    logits = mlm(**inputs).logits

            # Choose top k among non-special tokens
            logits[:, :, special_token_ids] = -float("inf")

            indices_in_mlm_tokens = (
                inputs.input_ids == mlm_tokenizer.mask_token_id
            ).nonzero(as_tuple=False) # if as_tuple=False, returns a tensor where column 1 indicates row indices, column 2 indicates column indices e.g. torch.Tensor([[0, 19],[0, 20], [0,38]])
            # print(f"-- location of masks including masks in the future spans: {indices_in_mlm_tokens}")
            
            # For each hypothesis in curr_full_text_hyp, first j mask locations are relevant
            indices_in_mlm_tokens = torch.cat([x[:j] for x in torch.chunk(indices_in_mlm_tokens, curr_queue_size)],dim=0)
            indices_in_mlm_tokens_0 = indices_in_mlm_tokens[:,0]
            indices_in_mlm_tokens_1 = indices_in_mlm_tokens[:,1]
            # print(f"-- location of masks (row indices): {indices_in_mlm_tokens_0}")
            # print(f"-- location of masks (col indices): {indices_in_mlm_tokens_1}")
            
            # Get top k tokens for the j masks
            predicted_token_ids = torch.topk(
                logits[indices_in_mlm_tokens_0, indices_in_mlm_tokens_1, :],
                k=config['k_per_location'],
                dim=-1,
            )            
            # print(f"-- top k tokens for each mask in each hypothesis: {predicted_token_ids.indices}")

            # When we do beam search, we only beam search up until the j masks.
            # print(f"-- shape of masked_sequence before slicing and padding: {masked_sequence.shape}")
            # print(f"-- masked_sequence before slicing and padding: {masked_sequence}")
            
            masked_sequence = [masked_sequence[ix, :indices_in_mlm_tokens_1[j*(ix+1)-1]+1] for ix in range(masked_sequence.shape[0])]
            masked_sequence = torch.nn.utils.rnn.pad_sequence(masked_sequence, batch_first=True, padding_value=mlm_tokenizer.pad_token_id)
            # print(f"-- shape of masked_sequence after slicing and padding: {masked_sequence.shape}")
            # print(f"-- masked_sequence after slicing and padding: {masked_sequence}")
            
            partial_hypotheses = get_beam_hypotheses_v0(source_text, 
                                masked_sequence, 
                                (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                predicted_token_ids.indices,
                                mlm_tokenizer, 
                                lossfns,
                                config)
            
            # print(f"-- num of returned partial hypotheses after beam search: {len(partial_hypotheses)}")
            # print(f"-- returned partial hypotheses after beam search: {partial_hypotheses}")
            
            partial_hypotheses = sum(partial_hypotheses, [])
            
            # print(f"-- num of partial hypotheses after unraveling disregarding the grouping by initial hypothesis: {len(partial_hypotheses)}")
            # print(f"-- partial hypotheses after unraveling disregarding the grouping by initial hypothesis: {partial_hypotheses}")
            # Extend the partial hypotheses to hypotheses pool for current mask span
            hypotheses.append(partial_hypotheses)

        # print("   * all mask length explored")
        hypotheses_all = sum(hypotheses, [])
        # print(f"-- # of hypotheses for current ({i}th) span: {len(hypotheses_all)}")
        # print(f"-- entire list of hypotheses for current ({i}th) span: {hypotheses_all}")
        
        # # Append snippet of text after current span and before next span before scoring
        if i < len(mask_spans) -1 :
            hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]] for x in hypotheses_all]
        else:
            hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:] for x in hypotheses_all]
        # print(f"-- entire list of hypotheses for current ({i}th) span for scoring: {hypotheses_all}") 
        
        # Scoring the hypotheses and select top beam hypotheses
        loss_weights = config['loss_weights']
        curr_loss = torch.zeros(len(hypotheses_all)).to(config['device'])
        data_loader = DataLoader(CustomDataset(hypotheses_all),batch_size=batch_size)

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
        
        torch.cuda.empty_cache()
        top_beams = torch.topk(curr_loss, k=config['beam_size'], dim=-1, largest=False).indices
        # print(f"-- selected {config['beam_size']} hypotheses' indices: {top_beams}")
        # print(f"-- selected {config['beam_size']} hypotheses for current ({i}th) span: {[hypotheses_all[ix] for ix in top_beams]}")
        queue = [hypotheses_all[ix] for ix in top_beams]
        del hypotheses_all
        # print(f"-- queue after extending current step's selected hypotheses: {queue}")
        
    final_hypotheses, new_best_weighted_loss_, new_best_allsat_, new_best_logging_loss_ = final_reranking(source_text,
                                                                                                        [list(queue)],
                                                                                                        lossfns,
                                                                                                        config,
                                                                                                        batch_size=32)
    return final_hypotheses, new_best_weighted_loss_, new_best_allsat_, new_best_logging_loss_

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

        # loss_weights = [1 - config['closs_weight'], config['closs_weight']]
        loss_weights = config['loss_weights']
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
                    batch_size:int=64,
                    main_constraint_loss_id:int=1) -> Tuple[List[str],torch.FloatTensor,torch.BoolTensor,torch.FloatTensor]:
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
    
    final_hypotheses = []
    best_weighted_loss = []
    best_allsat = []
    best_logging_loss = []
    
    # loss_weights = [1 - config['closs_weight'], config['closs_weight']]
    loss_weights = config['loss_weights']
    # for i in tqdm(range(len(hypotheses))):
    for i in range(len(hypotheses)):
        curr_loss = torch.zeros(len(hypotheses[i])).to(config['device'])
        logging_loss = torch.zeros((len(hypotheses[i]),len(lossfns))).to(config['device'])
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
            
        allsat_ix = torch.where(logging_loss[:,main_constraint_loss_id]< -math.log(config["min_epsilons"][0]))[0]
        if (len(allsat_ix) > 0) and (config['selection_criteria'] == "allsat_primary"):
        #if (allsat_ix.shape[0] > 0) and (config['selection_criteria'] == "allsat_primary"):
            # best_ix = allsat_ix[curr_loss[allsat_ix].argmin()]
            best_ix = allsat_ix[logging_loss[allsat_ix,0].argmin()]
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

