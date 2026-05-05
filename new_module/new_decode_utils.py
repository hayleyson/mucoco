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

from new_module.set_consistency_energy.locate_and_edit.edit import mask_text
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

def repeat_interleave_unravel(arr,split_blocks):
    arr_ = torch.split(arr.T,1,dim=1)
    arr_ = [x.repeat(1,split_blocks[i]).reshape(-1,1) for i,x in enumerate(arr_)]
    arr_ = torch.cat(arr_,dim=0)
    return arr_

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

def _scale_threshold_to_energy(threshold: float, threshold_scale: str="probability") -> float:
    if threshold_scale == "probability":
        return -math.log(threshold)
    elif threshold_scale == "energy":
        return threshold
    else:
        raise ValueError(f"Invalid threshold scale: {threshold_scale}")
    

def compute_allsat_from_thresholds(logging_loss: torch.Tensor, thresholds: List[float], threshold_scales: List[str]) -> torch.Tensor:
    """Per-sample AND over gated losses: ``logging_loss[:, i+1] < _scale_threshold_to_energy(thresholds[i], threshold_scales[i])``.

    Column 0 of ``logging_loss`` is not gated. Expect ``len(thresholds) == logging_loss.shape[1] - 1``.
    """
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    allsat = None
    for eps_idx, eps in enumerate(thresholds):
        allsat_i = logging_loss[:, eps_idx + 1] < _scale_threshold_to_energy(eps, threshold_scales[eps_idx])
        allsat = allsat_i if eps_idx == 0 else (allsat & allsat_i)
    return allsat


def get_beam_hypotheses_v0_variable_length_v2(source_text:str, 
                    masked_sequence:torch.Tensor, 
                    indices_in_mlm_tokens:Tuple[torch.Tensor],
                    predicted_token_ids:torch.Tensor,
                    mlm_tokenizer:transformers.AutoTokenizer, 
                    lossfns:List[lossbuilder.BaseLoss],
                    config:dict,
                    return_all_hypotheses:bool=False,
                    batch_size:int=16) -> List[List[str]]:
    
    # 변수 초기화
    final_hypotheses = [[] for i in range(len(masked_sequence))]
    final_hypotheses_losses = [[] for i in range(len(masked_sequence))]
    hypotheses = list(torch.split(masked_sequence,1,dim=0)) ## [torch.tensor([[a],[b],[c]]), torch.tensor([[d]])]
    edit_indices = sorted(list(set(indices_in_mlm_tokens[1].tolist())))
    loss_weights = config['loss_weights']

    if not return_all_hypotheses: ## 밖에서 reranking을 하는 경우가 아니라면 deletion 케이스도 처리
        # deletion 케이스 처리 : deletion case로 final_hypotheses, final_hypotheses_losses 초기화 
        first_mask_token_indices = [indices_in_mlm_tokens[1][indices_in_mlm_tokens[0]==i][0] for i in range(len(hypotheses))]
        final_hypotheses = [[hypotheses[i][:, :first_mask_token_indices[i]].squeeze(0)] for i in range(len(hypotheses))] # List[List[torch.tensor]] # 계속해서 누적될 hypotheses 목록
        tmp_hypotheses = [x[0].tolist() for x in final_hypotheses] # List[List[int]] # 이번 j 에서 고려할 hypotheses
        # print(f"{tmp_hypotheses}")

        curr_loss = torch.zeros(len(tmp_hypotheses)).to(config['device'])
        tmp_hypotheses_dec = mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True)
        data_loader = DataLoader(CustomDataset(tmp_hypotheses_dec),batch_size=batch_size)
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
            
        final_hypotheses_losses = [[x.squeeze(0)] for x in torch.split(curr_loss, 1)]
        # print(f"final_hypotheses_losses: {final_hypotheses_losses}")
        
    # variable replacement 케이스 처리
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
            
        curr_loss = torch.zeros(len(tmp_hypotheses)).to(config['device'])
        tmp_hypotheses_dec = mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True)
        data_loader = DataLoader(CustomDataset(tmp_hypotheses_dec),batch_size=batch_size)
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
            
        curr_loss = torch.split(curr_loss, num_initial_tmp_hypotheses, dim=0)
        top_beams = [torch.topk(x, k=config['beam_size'], dim=-1, largest=False).indices for x in curr_loss]
        tmp_hypotheses = torch.split(tmp_hypotheses, num_initial_tmp_hypotheses, dim=0)
        for jx, ix in enumerate(batch_ids_to_edit):
            hypotheses[ix] = torch.cat([tmp_hypotheses[jx][top_beams[jx]], masked_sequence[ix][curr_edit_index+1:].unsqueeze(0).repeat(config['beam_size'],1)], dim=-1)
            final_hypotheses[ix].extend(tmp_hypotheses[jx][top_beams[jx]])
            final_hypotheses_losses[ix].extend(curr_loss[jx][top_beams[jx]])

    if return_all_hypotheses:
        dec_final_hypotheses = [mlm_tokenizer.batch_decode(x, skip_special_tokens=True) for x in final_hypotheses]
        dec_final_hypotheses = sum(dec_final_hypotheses,[])
        return dec_final_hypotheses, final_hypotheses_losses
    else:
        final_top_beams = [torch.topk(torch.stack(x), k=config['beam_size'], dim=-1, largest=False).indices for x in final_hypotheses_losses]
        final_final_hypotheses = [[x[i] for i in y] for x,y in zip(final_hypotheses,final_top_beams)]
        final_final_scores = [[x[i] for i in y] for x,y in zip(final_hypotheses_losses,final_top_beams)]

        return [mlm_tokenizer.batch_decode(x, skip_special_tokens=True) for x in final_final_hypotheses], final_final_scores
        
def editing_with_delete_variable_replace(source_text:str, test_sent:str, test_sent_span_lengths:List[int], 
                                         mlm:AutoModelForMaskedLM, mlm_tokenizer:AutoTokenizer, 
                                         lossfns:List[lossbuilder.BaseLoss], config: dict, batch_size:int=16) -> \
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
        best_allsat: torch.ByteTensor of indicator(1,0) whether the best hypothesis satisfy cutoff (thresholds) for constraint energy score.
        best_logging_loss: torch.FloatTensor of shape (num samples, 2) of fluency energy score and constraint energy score for each best hypothesis.
    """
    
    # merge masks
    test_sent_merged = re.sub(r"(<mask>)+", "<mask>", test_sent)
    
    # Max number of mask tokens to replace each span
    # max_mask_cnt_per_span = [max(x, config['max_tokens_per_span']) for x in test_sent_span_lengths]
    max_mask_cnt_per_span = [config['max_tokens_per_span'] for x in test_sent_span_lengths]

    # Get the span information of merged masks in the test sentence
    mask_spans = [x.span() for x in re.finditer('<mask>',test_sent_merged)]

    special_token_ids = mlm_tokenizer.convert_tokens_to_ids(mlm_tokenizer.all_special_tokens)

    queue = []
    queue.append(test_sent_merged[:mask_spans[0][0]])
    for i in range(len(mask_spans)):
        curr_queue_size = len(queue)
        
        # candidate generation
        curr_full_text_hyp = [base_hyp + "<mask>" * max_mask_cnt_per_span[i] + test_sent_merged[mask_spans[i][1]:] for base_hyp in queue]
        ## Tokenize & conduct MLM inference
        inputs = mlm_tokenizer(
            curr_full_text_hyp, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        ) ## add_special_tokens=False to skip adding bos token
        inputs = inputs.to(config['device']) 
        masked_sequence=inputs['input_ids']
        
        if config['consider_prompt_for_cand_gen']:
            
            prompt_enc=mlm_tokenizer(mlm_tokenizer.bos_token + source_text,add_special_tokens=False, return_tensors="pt", padding=True, truncation=True).to(config['device'])
            prompt_enc['input_ids']=prompt_enc['input_ids'].expand(curr_queue_size,-1)
            prompt_enc['attention_mask']=prompt_enc['attention_mask'].expand(curr_queue_size,-1)
            
            input_tokens = torch.cat([prompt_enc.input_ids, inputs.input_ids], dim=1).to(config['device'])
            attention_masks = torch.cat([prompt_enc.attention_mask, inputs.attention_mask], dim=1).to(config['device'])
            
            with torch.no_grad():
                logits = mlm(input_ids = input_tokens, 
                            attention_mask = attention_masks).logits

            # Choose top k among non-special tokens
            logits = logits[:, prompt_enc.input_ids.shape[1]:]
            
        else:
            with torch.no_grad():
                logits = mlm(**inputs).logits
        
        ## Choose top k among non-special tokens
        logits[:, :, special_token_ids] = -float("inf")

        indices_in_mlm_tokens = (
            inputs.input_ids == mlm_tokenizer.mask_token_id
        ).nonzero(as_tuple=False) # if as_tuple=False, returns a tensor where column 1 indicates row indices, column 2 indicates column indices e.g. torch.Tensor([[0, 19],[0, 20], [0,38]])
        
        ## get post context 
        if i == len(mask_spans) -1:
            post_context = test_sent_merged[mask_spans[i][1]:]
        else:
            post_context = test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]]

        ## For each hypothesis in curr_full_text_hyp, first max_mask_cnt_per_span[i] mask locations are relevant
        indices_in_mlm_tokens = torch.cat([x[:max_mask_cnt_per_span[i]] for x in torch.chunk(indices_in_mlm_tokens, curr_queue_size)],dim=0)
        indices_in_mlm_tokens_0 = indices_in_mlm_tokens[:,0]
        indices_in_mlm_tokens_1 = indices_in_mlm_tokens[:,1]

        ## Get top k tokens for the j masks
        predicted_token_ids = torch.topk(
            logits[indices_in_mlm_tokens_0, indices_in_mlm_tokens_1, :],
            k=config['k_per_location'],
            dim=-1,
        )            

        ## beam search에 넣기 전에 이런 작업을 해주는게 좋을까? ## right side를 아예 안볼거면 ok.  -> 꼭 해주지 않아도 indices_in_mlm_tokens 에서 현재 span까지만 index를 뽑기 때문에 같은 결과가 나오긴 함.
        masked_sequence = [masked_sequence[ix, :indices_in_mlm_tokens_1[max_mask_cnt_per_span[i]*(ix+1)-1]+1] for ix in range(masked_sequence.shape[0])]
        masked_sequence = torch.nn.utils.rnn.pad_sequence(masked_sequence, batch_first=True, padding_value=mlm_tokenizer.pad_token_id)        

        hypotheses=list(queue) # deletion case
        beam_outputs, _ = get_beam_hypotheses_v0_variable_length_v2(source_text, 
                                masked_sequence, 
                                (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                predicted_token_ids.indices,
                                mlm_tokenizer, 
                                lossfns,
                                config,
                                return_all_hypotheses=True,
                                batch_size=batch_size)
        hypotheses.extend(beam_outputs)
        if i < len(mask_spans) -1 :
            hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]] for x in hypotheses]
        else:
            hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:] for x in hypotheses]

        # Scoring the hypotheses and select top beam hypotheses
        curr_loss = torch.zeros(len(hypotheses_all)).to(config['device'])
        data_loader = DataLoader(CustomDataset(hypotheses_all),batch_size=batch_size)
        logging_loss = torch.zeros((len(hypotheses_all),len(lossfns))).to(config['device'])

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
            curr_loss += config['loss_weights'][lossid] * lossvalue
            logging_loss[:, lossid] = lossvalue.clone()

        torch.cuda.empty_cache()
        if i == len(mask_spans) -1:
            allsat_mask = compute_allsat_from_thresholds(
                logging_loss, config["thresholds"], config["threshold_scales"]
            )
            allsat_ix = torch.where(allsat_mask)[0]
            if (allsat_ix.numel() > 0) and (config['selection_criteria'] == "allsat_primary"):
                best_ix = allsat_ix[logging_loss[allsat_ix, 0].argmin()]
            else: ## in case config['selection_criteria'] == "weighted_sum" or allsat is all False
                best_ix = torch.argmin(curr_loss)
            
            final_hypotheses = [hypotheses_all[best_ix]]
            best_weighted_loss = [curr_loss[best_ix].item()]
            best_allsat = [
                1
                if allsat_ix.numel() > 0 and (allsat_ix == best_ix).any().item()
                else 0
            ]
            best_logging_loss = [logging_loss[best_ix].cpu().tolist()]
        else:
            top_beams = torch.topk(curr_loss, k=config['beam_size'], dim=-1, largest=False).indices
            queue = [hypotheses_all[ix] for ix in top_beams]
            
        del curr_loss, logging_loss
        torch.cuda.empty_cache()

    return final_hypotheses, torch.FloatTensor(best_weighted_loss).to(config['device']), \
            torch.BoolTensor(best_allsat).to(config['device']), torch.FloatTensor(best_logging_loss).to(config['device'])



def get_beam_4sce(source_text:str, 
                    masked_sequence:torch.Tensor, 
                    indices_in_mlm_tokens:Tuple[torch.Tensor],
                    predicted_token_ids:torch.Tensor,
                    lefthand_instance_text:str,
                    post_context:str,
                    mlm_tokenizer:transformers.AutoTokenizer, 
                    lossfns:List[lossbuilder.BaseLoss],
                    config:dict,
                    batch_size:int=16) -> List[List[str]]:

    """
    Performs a beam search using pre-computed token-level candidates, returning 
    hypotheses with lengths up to the maximum permitted number of mask tokens.
    
    Args:
        source_text: Prefix text (unused in SC energy task, remains as a placeholder).
        masked_sequence: Sequence text up to the last <mask> token of the current span.
        indices_in_mlm_tokens: Indices of <mask> tokens in the MLM tokenizer.
        predicted_token_ids: Pre-computed token-level candidates.
        nonlocated_instances_text: Text of instances not currently being edited.
        mlm_tokenizer: The MLM tokenizer.
        lossfns: List of loss functions used for scoring.
        config: Configuration dictionary.
        batch_size: Batch size for processing.
    """
    
    # Initialize variables
    final_hypotheses = [[] for i in range(len(masked_sequence))]
    final_hypotheses_losses = [[] for i in range(len(masked_sequence))]
    hypotheses = list(torch.split(masked_sequence,1,dim=0)) ## [torch.tensor([[a],[b],[c]]), torch.tensor([[d]])]
    edit_indices = sorted(list(set(indices_in_mlm_tokens[1].tolist())))
    loss_weights = config['loss_weights']
    # logger.debug(f"edit_indices: {edit_indices}")
    for curr_edit_index in edit_indices:
        # logger.debug(f"-------- curr_edit_index: {curr_edit_index} -------")
        batch_ids_to_edit = indices_in_mlm_tokens[0][indices_in_mlm_tokens[1]==curr_edit_index].tolist()
        num_initial_hypotheses = [len(hypotheses[i]) for i in batch_ids_to_edit] ## keep track of initial hypotheses count e.g. [3, 1]
        tmp_hypotheses = [hypotheses[i].repeat((config['k_per_location'],1)) for i in batch_ids_to_edit] ## [torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c]]), torch.tensor([[d],[d],[d]])]
        num_initial_tmp_hypotheses = [len(x) for x in tmp_hypotheses]
        tmp_hypotheses = torch.cat(tmp_hypotheses,dim=0) ## torch.tensor([[a],[b],[c],[a],[b],[c],[a],[b],[c],[d],[d],[d]])
        new_func_candidates = predicted_token_ids[indices_in_mlm_tokens[1]==curr_edit_index] ## shape: (len(batch_ids_to_edit), k_per_location) e.g. [[x,y,z],[q,w,e]]
        new_func_candidates = repeat_interleave_unravel(new_func_candidates,num_initial_hypotheses) ## shape: (sum(num_initial_hypotheses), k_per_location) e.g. [[x],[x],[x],[y],[y],[y],[z],[z],[z],[q],[w],[e]]
        new_func_candidates = new_func_candidates.to(config['device'])
        # logger.debug(f"tmp_hypotheses: {tmp_hypotheses}")
        tmp_hypotheses = torch.cat((tmp_hypotheses[ :, :curr_edit_index], new_func_candidates),dim=-1) ## tmp_hypotheses: [(a,b,c),(a,b,c), ..., (a,b,c)], new_func_candidates: [(p,p,p), (q,q,q), ..., (v,v,v)]
        # logger.debug(f"tmp_hypotheses: {tmp_hypotheses}")
        
        curr_loss = torch.zeros(len(tmp_hypotheses)).to(config['device'])
        tmp_hypotheses_dec = mlm_tokenizer.batch_decode(tmp_hypotheses)
        # logger.debug(f"tmp_hypotheses_dec: {tmp_hypotheses_dec}")
        tmp_hypotheses_dec1 = tmp_hypotheses_dec
        tmp_hypotheses_dec2 = [lefthand_instance_text + hyp + post_context for hyp in tmp_hypotheses_dec]
        # logger.debug(f"tmp_hypotheses_dec2: {tmp_hypotheses_dec2}")
        
        data_loader1 = DataLoader(CustomDataset(tmp_hypotheses_dec1),batch_size=batch_size)
        for lossid, lossname in enumerate(config["losses"]):
            if lossid != 0:
                continue
            data_loader = data_loader1
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
            # logger.debug(f"{lossid}th loss: {lossvalue}")
            curr_loss += loss_weights[lossid] * lossvalue
            
        
        curr_loss = torch.split(curr_loss, num_initial_tmp_hypotheses, dim=0)
        top_beams = [torch.topk(x, k=config['beam_size'], dim=-1, largest=False).indices for x in curr_loss]
        # logger.debug(f"top_beams: {top_beams}")
        tmp_hypotheses = torch.split(tmp_hypotheses, num_initial_tmp_hypotheses, dim=0)
        for jx, ix in enumerate(batch_ids_to_edit):
            hypotheses[ix] = torch.cat([tmp_hypotheses[jx][top_beams[jx]], masked_sequence[ix][curr_edit_index+1:].unsqueeze(0).repeat(config['beam_size'],1)], dim=-1)
            final_hypotheses[ix].extend(tmp_hypotheses[jx][top_beams[jx]])
            final_hypotheses_losses[ix].extend(curr_loss[jx][top_beams[jx]])
        # logger.debug(f"hypotheses: {hypotheses}")
        # logger.debug(f"final_hypotheses: {final_hypotheses}")

    dec_final_hypotheses = [mlm_tokenizer.batch_decode(x) for x in final_hypotheses]
    dec_final_hypotheses = sum(dec_final_hypotheses,[])
    # logger.debug(f"dec_final_hypotheses: {dec_final_hypotheses}")
    return dec_final_hypotheses, final_hypotheses_losses
        
def editing_4sce(source_text:str, test_sent_orig:str, test_sent:str, test_sent_span_lengths:List[int], 
                located_instance_index: int, 
                mlm:AutoModelForMaskedLM, mlm_tokenizer:AutoTokenizer, 
                lossfns:List[lossbuilder.BaseLoss], config: dict, batch_size:int=16,
                post_context_mode: str = 'original') -> \
                    Tuple[List[str],torch.FloatTensor,torch.BoolTensor,torch.FloatTensor]:
    """
    Performs editing on a specific located instance (e.g., a QA pair in Set-LConVQA).
    
    Args:
        source_text: Prefix text.
        test_sent: The complete input text containing all instances, with masking applied to target spans within the located instance.
        test_sent_span_lengths: Number of tokens within each masked span of the instance.
        located_instance_index: The index of the instance being edited.
        mlm: The Masked Language Model (MLM) used for candidate generation.
        mlm_tokenizer: The tokenizer associated with the MLM.
        lossfns: A list of loss functions for energy evaluation.
        config: Configuration dictionary for hyperparameters and settings.
        batch_size: Batch size for model inference.
    

    NOTE: Energy calculation for Set Consistency (SC) Energy differs from other tasks.
    Since the input text is a collection of instances (e.g., QA pairs for Set-LConVQA), 
    fluency is evaluated only for the target ('located') instance in isolation, rather than including 
    preceding instances. Additionally, because the order of instances should not affect the energy value, 
    all non-located instances are moved before the located instance when evaluating set consistency. 
    This allows set consistency to be evaluated during inner beam search, which typically only considers 
    preceding context.
    """
    cls_token = lossfns[1].tokenizer.cls_token
    sep_token = "." # hard coding to match actual dataset format
    special_token_ids = mlm_tokenizer.convert_tokens_to_ids(mlm_tokenizer.all_special_tokens)
    
    
    def detect_instance(set_text: str, sep_token: str, cls_token: str):

        # set_text == text, e.g., '<s> qa pair 1 </s> qa pair 2 ... </s>
        if set_text.startswith(cls_token):
            out = set_text[len(cls_token):].split(sep_token)
        else:
            out = set_text.split(sep_token)
        
        if out[-1] == '':
            out = out[:-1]

        return [o+sep_token for o in out]


    def join_instances(instances: List[str]):

        return ''.join(instances)
        
    def get_post_contexts(test_sent_orig: str, test_sent: str, mode:str="original"):
        
        test_sent_tokens = mlm_tokenizer.tokenize(test_sent)
        test_sent_orig_tokens = mlm_tokenizer.tokenize(test_sent_orig)
        
        # Find positions where mask tokens end (transition from <mask> to non-mask)
        # and build a list of post contexts that follow each streak of mask tokens
        post_contexts = []
        for i in range(len(test_sent_tokens)):
            is_mask = (test_sent_tokens[i] == "<mask>")
            is_prev_mask = ((i > 0) and (test_sent_tokens[i - 1] == "<mask>"))
            
            if is_prev_mask and not is_mask:
                post_contexts.append(mlm_tokenizer.convert_tokens_to_string(test_sent_orig_tokens[i:]) if mode == "original" \
                                else mlm_tokenizer.convert_tokens_to_string(test_sent_tokens[i:]))
        
        # Handle case where sentence ends with mask tokens
        if test_sent_tokens and test_sent_tokens[-1] == "<mask>":
            post_contexts.append("")
        return post_contexts
    
    post_contexts = get_post_contexts(test_sent_orig, test_sent, mode=post_context_mode)
    
    # first test sentence into instances & set only the located instance as test sentence
    instances = detect_instance(test_sent, sep_token, cls_token)
    try:
        test_sent = instances[located_instance_index]
    except:
        logger.info(f"[ERROR] located_instance_index {located_instance_index} is out of range for {len(instances)} instances")
        logger.info(f"instances: {instances}")
        test_sent = ""
    if "<mask>" not in test_sent:
        test_sent = ""
        for instance in instances:
            if "<mask>" in instance:
                test_sent = instance
                located_instance_index = instances.index(instance)
                break

    lefthand_instances = instances[:located_instance_index] if located_instance_index > 0 else []
    lefthand_instance_text = join_instances(lefthand_instances)
    righthand_instances = instances[located_instance_index+1:] if located_instance_index+1 < len(instances) else []
    righthand_instance_text = join_instances(righthand_instances)

    
    # merge consecutive <mask> tokens in the test sentence
    test_sent_merged = re.sub(r"(<mask>)+", "<mask>", test_sent)
    

    # max_mask_cnt_per_span = [max(x, config['max_tokens_per_span']) for x in test_sent_span_lengths]
    max_mask_cnt_per_span = [config['max_tokens_per_span'] for x in test_sent_span_lengths]
    mask_spans = [x.span() for x in re.finditer('<mask>',test_sent_merged)]

    queue = []
    queue.append(test_sent_merged[:mask_spans[0][0]])
    for i in range(len(mask_spans)):
        curr_queue_size = len(queue)
        
        # Try all possible numbers of mask tokens from 1 up to max_mask_cnt_per_span
        mask_counts = [k for k in range(max_mask_cnt_per_span[i], max_mask_cnt_per_span[i] + 1)]
        # logger.debug(f"mask_counts: {mask_counts}")
        curr_full_text_hyp = []
        for k in mask_counts:
            curr_full_text_hyp.extend([base_hyp + "<mask>" * k + test_sent_merged[mask_spans[i][1]:] for base_hyp in queue])
        # logger.debug(f"curr_full_text_hyp: {curr_full_text_hyp}")

        # Total number of hypotheses being considered in this MLM call
        total_mlm_batch_size = len(curr_full_text_hyp)
        
        inputs = mlm_tokenizer(
            curr_full_text_hyp, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False # skip adding bos token here
        ).to(config['device']) 
        masked_sequence = inputs.input_ids.clone()
        # logger.debug(f'inputs.attention_mask: {inputs.attention_mask}')
        
        lefthand_instance_enc = mlm_tokenizer(
            mlm_tokenizer.bos_token + lefthand_instance_text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        ).to(config['device']) 
        lefthand_ids = lefthand_instance_enc['input_ids'][0]
        
        righthand_instance_enc = mlm_tokenizer(
            righthand_instance_text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        ).to(config['device']) 
        righthand_ids = righthand_instance_enc['input_ids'][0]

        input_tokens_list = []
        attention_mask_list = []
        for v_idx in range(total_mlm_batch_size):
            valid_len = inputs.attention_mask[v_idx].sum().item()
            valid_inputs = inputs.input_ids[v_idx, :valid_len]
            cat_ids = torch.cat([lefthand_ids, valid_inputs, righthand_ids], dim=0)
            input_tokens_list.append(cat_ids)
            attention_mask_list.append(torch.ones_like(cat_ids))

        input_tokens = torch.nn.utils.rnn.pad_sequence(input_tokens_list, batch_first=True, padding_value=mlm_tokenizer.pad_token_id).to(config['device'])
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(config['device'])
                
        # logger.debug(f"input_tokens: {input_tokens}")
        # logger.debug(f"attention_masks: {attention_masks}")

        # MLM inference happening here
        with torch.no_grad():
            logits = mlm(input_ids = input_tokens, 
                        attention_mask = attention_masks).logits

        logits[:, :, special_token_ids] = -float("inf")

        # Get indices of <mask> tokens for the current span
        indices_in_mlm_tokens = (
            inputs.input_ids == mlm_tokenizer.mask_token_id
        ).nonzero(as_tuple=False) 
        
        indices_in_mlm_tokens_0 = indices_in_mlm_tokens[:,0]
        indices_in_mlm_tokens_1 = indices_in_mlm_tokens[:,1]
        
        # Filter indices to only include the masks from the current span (the first k masks)
        valid_mask_filter = []
        last_mask_indices = []
        for v_idx in range(total_mlm_batch_size):
            k = mask_counts[v_idx // curr_queue_size]
            row_mask_positions = (indices_in_mlm_tokens_0 == v_idx).nonzero(as_tuple=True)[0]
            valid_mask_filter.append(row_mask_positions[:k])
            
            row_masks = indices_in_mlm_tokens_1[indices_in_mlm_tokens_0 == v_idx]
            last_mask_indices.append(row_masks[k - 1].item())
        
        valid_mask_filter = torch.cat(valid_mask_filter)
        indices_in_mlm_tokens_0 = indices_in_mlm_tokens_0[valid_mask_filter]
        indices_in_mlm_tokens_1 = indices_in_mlm_tokens_1[valid_mask_filter]
        
        shifted_indices_1 = indices_in_mlm_tokens_1 + lefthand_ids.shape[0]
        
        
        # Get top k tokens for each <mask> token
        # Note: different samples in the batch have different mask counts now.
        predicted_token_ids = torch.topk(
            logits[indices_in_mlm_tokens_0, shifted_indices_1, :],
            k=config['k_per_location'],
            dim=-1,
        )            
        
        # logger.debug(f"predicted_token_ids: {predicted_token_ids}")
        masked_sequence = [masked_sequence[ix, :last_mask_indices[ix]+1] for ix in range(total_mlm_batch_size)]
        masked_sequence = torch.nn.utils.rnn.pad_sequence(masked_sequence, batch_first=True, padding_value=mlm_tokenizer.pad_token_id)        

        # logger.debug(f"masked_sequence: {masked_sequence}")
        # -------------------------------------------------------------------------
        # Generate sequence hypotheses by conducting beam search with generated token-level candidates
        # -------------------------------------------------------------------------
        # Include the deletion case as a baseline hypothesis (no tokens added for the current mask span).
        hypotheses=list(queue) 
        # Perform beam search to generate candidates with variable replacement lengths.
        beam_outputs, _ = get_beam_4sce(source_text, 
                                masked_sequence, 
                                (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                predicted_token_ids.indices,
                                lefthand_instance_text, post_contexts[i], 
                                mlm_tokenizer, 
                                lossfns,
                                config,
                                batch_size=batch_size)
        hypotheses.extend(beam_outputs)
        # logger.debug(f"beam_outputs: {beam_outputs}")
        
        # -------------------------------------------------------------------------
        # Scoring the hypotheses and select beam_size of hypotheses with lowest energy
        # -------------------------------------------------------------------------
        curr_loss = torch.zeros(len(hypotheses)).to(config['device'])
        logging_loss = torch.zeros((len(hypotheses),len(lossfns))).to(config['device'])

        # Set two data loaders for fluency energy and set consistency energy
        # for consistency, consider both side contexts (for post context, masks are filled with the original tokens.)
        hypotheses_with_bothside_contexts = [lefthand_instance_text + x + post_contexts[i] for x in hypotheses] 
        # logger.debug(f"hypotheses_with_bothside_contexts: {hypotheses_with_bothside_contexts}")
        
        data_loader2 = DataLoader(CustomDataset(hypotheses_with_bothside_contexts),batch_size=batch_size)

        # for fluency, only consider current instance and up to right before the next mask span
        if i < len(mask_spans) -1 :
            hypotheses = [x + test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]] for x in hypotheses]
        else:
            hypotheses = [x + test_sent_merged[mask_spans[i][1]:] for x in hypotheses]
        # logger.debug(f"hypotheses: {hypotheses}")
        data_loader1 = DataLoader(CustomDataset(hypotheses),batch_size=batch_size)
        
        for lossid, lossname in enumerate(config["losses"]):
            if lossid == 0: 
                data_loader = data_loader1
            else:
                data_loader = data_loader2
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
            curr_loss += config['loss_weights'][lossid] * lossvalue
            logging_loss[:, lossid] = lossvalue.clone()
        torch.cuda.empty_cache()
        # logger.debug(f"logging_loss: {logging_loss}")
        # If it is the last mask span, select the best hypothesis
        if i == len(mask_spans) -1:
            allsat_mask = compute_allsat_from_thresholds(
                logging_loss, config["thresholds"], config["threshold_scales"]
            )
            allsat_ix = torch.where(allsat_mask)[0]
            if (allsat_ix.numel() > 0) and (config['selection_criteria'] == "allsat_primary"):
                # logger.debug(f"Selected based on allsat_primary")
                best_ix = allsat_ix[logging_loss[allsat_ix, 0].argmin()]
            else: # in case config['selection_criteria'] == "weighted_sum" or allsat is all False
                # logger.debug(f"Selected based on weighted_sum")
                best_ix = torch.argmin(curr_loss)
            
            # replace located instance with the best hypothesis and join back instances to get the final hypothesis
            instances[located_instance_index] = hypotheses[best_ix]
            final_hypotheses = [join_instances(instances)]
            # logger.debug(f"final_hypotheses: {final_hypotheses}")
            best_weighted_loss = [curr_loss[best_ix].item()]
            best_allsat = [
                1
                if allsat_ix.numel() > 0 and (allsat_ix == best_ix).any().item()
                else 0
            ]
            best_logging_loss = [logging_loss[best_ix].cpu().tolist()]
        else:
            # If it is not the last mask span, update the queue with the beam size of hypotheses with lowest energy
            top_beams = torch.topk(curr_loss, k=config['beam_size'], dim=-1, largest=False).indices
            # Append the static text between the current and next mask spans for the next iteration.
            queue = [hypotheses[ix] for ix in top_beams]
        
        del curr_loss, logging_loss
        torch.cuda.empty_cache()

    return final_hypotheses, torch.FloatTensor(best_weighted_loss).to(config['device']), \
            torch.BoolTensor(best_allsat).to(config['device']), torch.FloatTensor(best_logging_loss).to(config['device'])


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
        best_allsat: torch.ByteTensor of indicator(1,0) whether the best hypotheses satisfy cutoff (thresholds) for constraint energy score.
        best_logging_loss: torch.FloatTensor of shape (num samples, 2) of fluency energy score and constraint energy score for each best hypothesis.
    """
    
    final_hypotheses = []
    best_weighted_loss = []
    best_allsat = []
    best_logging_loss = []
    
    loss_weights = config['loss_weights']
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
            
        allsat_mask = compute_allsat_from_thresholds(
            logging_loss, config["thresholds"], config["threshold_scales"]
        )
        allsat_ix = torch.where(allsat_mask)[0]
        if (allsat_ix.numel() > 0) and (config['selection_criteria'] == "allsat_primary"):
            best_ix = allsat_ix[logging_loss[allsat_ix, 0].argmin()]
        else: ## in case config['selection_criteria'] == "weighted_sum" or allsat is all False
            best_ix = torch.argmin(curr_loss)

        final_hypotheses.append(hypotheses[i][best_ix])
        best_weighted_loss.append(curr_loss[best_ix].item())
        best_allsat.append(
            1
            if allsat_ix.numel() > 0 and (allsat_ix == best_ix).any().item()
            else 0
        )
        best_logging_loss.append(logging_loss[best_ix].cpu().tolist())
    
        del curr_loss, logging_loss
        torch.cuda.empty_cache()
    return final_hypotheses, torch.FloatTensor(best_weighted_loss).to(config['device']), \
            torch.BoolTensor(best_allsat).to(config['device']), torch.FloatTensor(best_logging_loss).to(config['device'])
