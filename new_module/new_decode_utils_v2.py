"""
editing_with_delete_variable_replace 함수 업데이트를 위한 프로토타입,테스트 코드
"""

import random
import time
import re
from collections import defaultdict
import json 
from typing import List, Tuple
from copy import deepcopy

import tqdm
from torch.utils.data import DataLoader,Dataset
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import numpy as np
import pandas as pd
import wandb

from new_module.new_decode_utils import get_beam_hypotheses_v0, get_beam_hypotheses_v1, get_combi_hypotheses, final_reranking, analyze_span_lengths_and_count, editing_with_delete_variable_replace

import new_module.losses as lossbuilder

# util 함수 선언

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


def get_beam_hypotheses_v0_variable_length_v2(source_text:str, 
                    masked_sequence:torch.Tensor, 
                    indices_in_mlm_tokens:Tuple[torch.Tensor],
                    predicted_token_ids:torch.Tensor,
                    mlm_tokenizer:transformers.AutoTokenizer, 
                    lossfns:List[lossbuilder.BaseLoss],
                    config:dict,
                    return_all_hypotheses:bool=False,
                    primary_loss_only:bool=False) -> List[List[str]]:
    
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
        for lossid, lossname in enumerate(config["losses"]):
            with torch.no_grad():
                lossvalue = lossfns[lossid].compute_gold_loss(
                    source_text, mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True),
                    label_id=config['target_label_ids'][lossid],
                )
            torch.cuda.empty_cache()
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
        curr_loss = torch.zeros(tmp_hypotheses.shape[0]).to(config['device'])
        
        for lossid, lossname in enumerate(config["losses"]):
            if (primary_loss_only) and (lossid >0):
                break
                
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
            final_hypotheses[ix].extend(tmp_hypotheses[jx][top_beams[jx]])
            final_hypotheses_losses[ix].extend(curr_loss[jx][top_beams[jx]])

    if return_all_hypotheses:
        return [mlm_tokenizer.batch_decode(x, skip_special_tokens=True) for x in final_hypotheses], final_hypotheses_losses
    else:
        final_top_beams = [torch.topk(torch.stack(x), k=config['beam_size'], dim=-1, largest=False).indices for x in final_hypotheses_losses]
        final_final_hypotheses = [[x[i] for i in y] for x,y in zip(final_hypotheses,final_top_beams)]
        final_final_scores = [[x[i] for i in y] for x,y in zip(final_hypotheses_losses,final_top_beams)]

        return [mlm_tokenizer.batch_decode(x, skip_special_tokens=True) for x in final_final_hypotheses], final_final_scores
        
def get_beam_hypotheses_v0_variable_length_considering_post_context_v2(source_text:str, 
                    masked_sequence:torch.Tensor, 
                    post_context:str,
                    indices_in_mlm_tokens:Tuple[torch.Tensor],
                    predicted_token_ids:torch.Tensor,
                    mlm_tokenizer:transformers.AutoTokenizer, 
                    lossfns:List[lossbuilder.BaseLoss],
                    config:dict,
                    return_all_hypotheses:bool=False,
                    primary_loss_only:bool=False) -> List[List[str]]:
    
    # 변수 초기화
    # final_hypotheses = [[] for i in range(len(masked_sequence))]
    # final_hypotheses_losses = [[] for i in range(len(masked_sequence))]
    hypotheses = list(torch.split(masked_sequence,1,dim=0)) ## [torch.tensor([[a],[b],[c]]), torch.tensor([[d]])]
    edit_indices = sorted(list(set(indices_in_mlm_tokens[1].tolist())))
    loss_weights = config['loss_weights']

    # deletion 케이스 처리 : deletion case로 final_hypotheses, final_hypotheses_losses 초기화 
    first_mask_token_indices = [indices_in_mlm_tokens[1][indices_in_mlm_tokens[0]==i][0] for i in range(len(hypotheses))]
    final_hypotheses = [[hypotheses[i][:, :first_mask_token_indices[i]].squeeze(0)] for i in range(len(hypotheses))] # List[List[torch.tensor]] # 계속해서 누적될 hypotheses 목록
    tmp_hypotheses = [x[0].tolist() for x in final_hypotheses] # List[List[int]] # 이번 j 에서 고려할 hypotheses
    # print(f"{tmp_hypotheses}")

    curr_loss = torch.zeros(len(tmp_hypotheses)).to(config['device'])
    for lossid, lossname in enumerate(config["losses"]):
        with torch.no_grad():
            lossvalue = lossfns[lossid].compute_gold_loss(
                source_text, [x+post_context for x in mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True)],
                label_id=config['target_label_ids'][lossid],
            )
        torch.cuda.empty_cache()
        curr_loss += loss_weights[lossid] * lossvalue
            
    final_hypotheses_losses = [[x.squeeze(0)] for x in torch.split(curr_loss, 1)]
    # print(f"final_hypotheses_losses: {final_hypotheses_losses}")

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
        
        curr_loss = torch.zeros(tmp_hypotheses.shape[0]).to(config['device'])
        
        for lossid, lossname in enumerate(config["losses"]):
            
            if (primary_loss_only) and (lossid >0):
                break
            with torch.no_grad():
                lossvalue = lossfns[lossid].compute_gold_loss(
                    source_text, [x+post_context for x in mlm_tokenizer.batch_decode(tmp_hypotheses,skip_special_tokens=True)], ## consider post_context
                    label_id=config['target_label_ids'][lossid],
                )
            torch.cuda.empty_cache()
            curr_loss += loss_weights[lossid] * lossvalue
        curr_loss = torch.split(curr_loss, num_initial_tmp_hypotheses, dim=0)
        top_beams = [torch.topk(x, k=config['beam_size'], dim=-1, largest=False).indices for x in curr_loss]
        tmp_hypotheses = torch.split(tmp_hypotheses, num_initial_tmp_hypotheses, dim=0)
        for jx, ix in enumerate(batch_ids_to_edit):
            hypotheses[ix] = torch.cat([tmp_hypotheses[jx][top_beams[jx]], masked_sequence[ix][curr_edit_index+1:].unsqueeze(0).repeat(config['beam_size'],1)], dim=-1)
            final_hypotheses[ix].extend(tmp_hypotheses[jx][top_beams[jx]])
            final_hypotheses_losses[ix].extend(curr_loss[jx][top_beams[jx]])

    if return_all_hypotheses:
        return [[y+post_context for y in mlm_tokenizer.batch_decode(x, skip_special_tokens=True)] for x in final_hypotheses], final_hypotheses_losses
    else:
        final_top_beams = [torch.topk(torch.stack(x), k=config['beam_size'], dim=-1, largest=False).indices for x in final_hypotheses_losses]
        final_final_hypotheses = [[x[i] for i in y] for x,y in zip(final_hypotheses,final_top_beams)]
        final_final_scores = [[x[i] for i in y] for x,y in zip(final_hypotheses_losses,final_top_beams)]

        return [[y+post_context for y in mlm_tokenizer.batch_decode(x, skip_special_tokens=True)] for x in final_final_hypotheses], final_final_scores

# prototype위한 변수 세팅
run_path = 'hayleyson/toxicity-decoding/bx3p1fwj'
api = wandb.Api()
run = api.run(run_path)
config = run.config
# config['model_paths'][0] = 'gpt2-large'
# config['tokenizer_paths'][0] = 'gpt2-large'

class dummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

build_loss_args = dummyArgs(**config["build_loss_dict"])
build_loss_args.task = config["task"]

mlm = AutoModelForMaskedLM.from_pretrained('roberta-large').to(config['device'])
mlm_tokenizer = AutoTokenizer.from_pretrained('roberta-large')

## load tokenizer, models, define losses
name2tokenizer = {}
name2model = {}
name2config = {}
loss2tokenizer = {}
embed_luts = []

for i, model_path in enumerate(config["model_paths"]):
    if (
        model_path not in name2model
    ):  # making sure we are not loading the model twice in case some constraints use the same model.
        try:
            name2tokenizer[config["tokenizer_paths"][i]] = AutoTokenizer.from_pretrained(
                config["tokenizer_paths"][i],
                cache_dir=config["cache_dir"],
                use_fast=True,
            )
        except:
            name2tokenizer[config["tokenizer_paths"][i]] = AutoTokenizer.from_pretrained(
                config["tokenizer_paths"][i],
                cache_dir=config["cache_dir"],
                use_fast=False,
            )

        name2config[model_path] = AutoConfig.from_pretrained(
            model_path, cache_dir=config["cache_dir"]
        )

        if config["model_types"][i] == "RobertaCustomForSequenceClassification":
            pass
        else:
            name2model[model_path] = lossbuilder.ModelWrapper(
                getattr(transformers, config["model_types"][i]).from_pretrained(
                    model_path,
                    config=name2config[model_path],
                    cache_dir=config["cache_dir"],
                )
            )
        name2model[model_path].eval()
        name2model[model_path].to(config['device'])

lossfns = []
for i, loss in enumerate(config["losses"]):
    lossfns.append(
        lossbuilder.build_loss(
            loss,
            name2model[config["model_paths"][i]],
            name2tokenizer[config["tokenizer_paths"][i]],
            build_loss_args,
        )
    )
    lossfns[i].tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})
    loss2tokenizer[loss] = lossfns[i].tokenizer

special_token_ids = mlm_tokenizer.convert_tokens_to_ids(mlm_tokenizer.all_special_tokens)
intermediate_outputs = pd.read_json('outputs/toxicity/llm/bx3p1fwj/outputs_epsilon0.9.txt.intermediate', lines=True)
intermediate_outputs = intermediate_outputs.explode('generations').reset_index(drop=True)
intermediate_outputs_eda = intermediate_outputs.loc[intermediate_outputs['generations'].apply(len) != 0].reset_index(drop=True)
intermediate_outputs_eda['prompt'] = intermediate_outputs_eda['prompt'].apply(lambda x: x['text'])
intermediate_outputs_eda['masked_sentences'] = intermediate_outputs_eda['generations'].apply(lambda x: [item[1] for item in x.items() if 'mask' in item[0]])
# intermediate_outputs_eda = intermediate_outputs_eda.explode('masked_sentences').reset_index(drop=True)
intermediate_outputs_eda['masked_sentences'] = intermediate_outputs_eda['masked_sentences'].apply(lambda x: x[0])
print(f"Number of total samples: {len(intermediate_outputs_eda)}")
all_source_texts = intermediate_outputs_eda['prompt'].tolist()
all_masked_sentences = intermediate_outputs_eda['masked_sentences'].tolist()
mask_info_dicts = []
span_lengths_es = []

for test_sent in all_masked_sentences:
    
    mask_info_dict, span_lengths = analyze_span_lengths_and_count(test_sent)
    mask_info_dicts.append(mask_info_dict)
    span_lengths_es.append(span_lengths)
    
# 각 방안 별로 결과가 어떻게 달라지는지 보다 여러 샘플에서 확인하기 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str)
args = parser.parse_args()

torch.cuda.empty_cache()
method = args.method
start = time.time()
results = []


# 1) span 돌면서 decoding
# 2) final로는 1개의 candidate만 나옴 
## editing_with_delete_variable_replace
## return 되어야 할 값들 목록 : final_hypotheses, new_best_weighted_loss_, new_best_allsat_, new_best_logging_loss_

random.seed(999)
idxes_for_test = random.sample(range(len(all_source_texts)),50)
print(f"Number of samples: {len(idxes_for_test)}")

batch_size = 64
loss_weights = config['loss_weights']

for sample_idx in tqdm.tqdm(idxes_for_test[:]):
     
    source_text = all_source_texts[sample_idx]
    test_sent = all_masked_sentences[sample_idx]
    test_sent_span_lengths = span_lengths_es[sample_idx]

    if method == "0":
        final_hypotheses_curr, new_best_weighted_loss_curr, new_best_allsat_curr, new_best_logging_loss_curr = \
                editing_with_delete_variable_replace(source_text, test_sent, test_sent_span_lengths, mlm, mlm_tokenizer, lossfns, config)
        results.append(final_hypotheses_curr)
    else:
        # merge masks
        test_sent_merged = re.sub(r"(<mask>)+", "<mask>", test_sent)

        # Max number of mask tokens to replace each span
        max_mask_cnt_per_span = [max(x, config['max_tokens_per_span']) for x in test_sent_span_lengths]

        # Get the span information of merged masks in the test sentence
        mask_spans = [x.span() for x in re.finditer('<mask>',test_sent_merged)]

        queue = []
        queue.append(test_sent_merged[:mask_spans[0][0]])
        for i in range(len(mask_spans)):
            curr_queue_size = len(queue)

            # candidate generation
            curr_full_text_hyp = [base_hyp + "<mask>" * max_mask_cnt_per_span[i] + test_sent_merged[mask_spans[i][1]:] for base_hyp in queue]
            ## Tokenize & conduct MLM inference
            inputs = mlm_tokenizer(
                curr_full_text_hyp, return_tensors="pt", padding=True, truncation=True
            )
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
            if method == "1":
                hypotheses=list(queue) # deletion case
                hypotheses.extend(get_beam_hypotheses_v0_variable_length_v2(source_text, 
                                        masked_sequence, 
                                        (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                        predicted_token_ids.indices,
                                        mlm_tokenizer, 
                                        lossfns,
                                        config,
                                        return_all_hypotheses=True)[0][0])
                # print(f"hypotheses: {hypotheses}")
                if i < len(mask_spans) -1 :
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]] for x in hypotheses]
                else:
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:] for x in hypotheses]

                # Scoring the hypotheses and select top beam hypotheses
                

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
                if i == len(mask_spans) -1:
                    top_beams = torch.topk(curr_loss, k=1, dim=-1, largest=False).indices
                    new_best_weighted_loss_ = curr_loss[top_beams]
                else:
                    top_beams = torch.topk(curr_loss, k=config['beam_size'], dim=-1, largest=False).indices
                
                queue = [hypotheses_all[ix] for ix in top_beams]
                # print(f"queue: {queue}")
                # for item in queue:
                #     print(item)
                # print('')
            elif method == "2":
                
                # hypotheses=list(queue) # deletion case
                # hypotheses.extend(get_beam_hypotheses_v0_variable_length(source_text, 
                hypotheses, scores = get_beam_hypotheses_v0_variable_length_v2(source_text, # deletion 케이스도 beam 안에서 고려
                                        masked_sequence, 
                                        (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                        predicted_token_ids.indices,
                                        mlm_tokenizer, 
                                        lossfns,
                                        config)
                hypotheses = hypotheses[0]
                # print(f"hypotheses: {hypotheses}")
                if i < len(mask_spans) -1 :
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]] for x in hypotheses]
                else:
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:] for x in hypotheses]
                
                if i == len(mask_spans) -1:
                    queue, new_best_weighted_loss_, new_best_allsat_, new_best_logging_loss_ = final_reranking(source_text,
                                                                                                                [hypotheses_all],
                                                                                                                lossfns,
                                                                                                                config,
                                                                                                                batch_size=32)
                else:
                    queue = hypotheses_all
                # print(f"queue: {queue}")
                # for item in queue:
                #     print(item)
                # print('')
                
            elif method == "3":
                
                tmp_config = deepcopy(config)
                if i == len(mask_spans) -1:
                    tmp_config['beam_size'] = 1
                
                # hypotheses=[x+post_context for x in list(queue)] # deletion case
                # hypotheses.extend(get_beam_hypotheses_v0_variable_length_considering_post_context(source_text, 
                hypotheses, scores = get_beam_hypotheses_v0_variable_length_considering_post_context_v2(source_text, 
                                        masked_sequence, 
                                        post_context,
                                        (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                        predicted_token_ids.indices,
                                        mlm_tokenizer, 
                                        lossfns,
                                        tmp_config)
                hypotheses = hypotheses[0]
                # print(f"hypotheses: {hypotheses}")
                
                # if i == len(mask_spans) -1:
                #     queue, new_best_weighted_loss_, new_best_allsat_, new_best_logging_loss_ = final_reranking(source_text,
                #                                                                                                 [hypotheses],
                #                                                                                                 lossfns,
                #                                                                                                 config,
                #                                                                                                 batch_size=32)
                # else:
                #     queue = hypotheses
                queue = hypotheses
                
                # print(f"queue: {queue}")
                # for item in queue:
                #     print(item)
                # print('')
                
            elif method == "3_2":
                
                # tmp_config = deepcopy(config)
                # if i == len(mask_spans) -1:
                #     tmp_config['beam_size'] = 1
                
                # hypotheses=[x+post_context for x in list(queue)] # deletion case
                # hypotheses.extend(get_beam_hypotheses_v0_variable_length_considering_post_context(source_text, 
                hypotheses, scores = get_beam_hypotheses_v0_variable_length_considering_post_context_v2(source_text, 
                                        masked_sequence, 
                                        post_context,
                                        (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                        predicted_token_ids.indices,
                                        mlm_tokenizer, 
                                        lossfns,
                                        config, 
                                        primary_loss_only=True)
                hypotheses = hypotheses[0]
                # print(f"hypotheses: {hypotheses}")
                if i == len(mask_spans) -1:
                    # Scoring the hypotheses and select top beam hypotheses
                    batch_size = 64

                    curr_loss = torch.zeros(len(hypotheses)).to(config['device'])
                    data_loader = DataLoader(CustomDataset(hypotheses),batch_size=batch_size)

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
                    top_beams = torch.topk(curr_loss, k=1, dim=-1, largest=False).indices
                    new_best_weighted_loss_ = curr_loss[top_beams]    
                    queue = [hypotheses[ix] for ix in top_beams]
                
                else:
                    queue = hypotheses
                    # print(f"queue: {queue}")
                    # for item in queue:
                    #     print(item)
                    # print('')
                
                
            elif method == "4" or method == "1_2":
                hypotheses=list(queue) # deletion case
                hypotheses.extend(get_beam_hypotheses_v0_variable_length_v2(source_text, 
                                        masked_sequence, 
                                        (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                        predicted_token_ids.indices,
                                        mlm_tokenizer, 
                                        lossfns,
                                        config,
                                        return_all_hypotheses=True,
                                        primary_loss_only=True)[0][0])
                # print(f"hypotheses: {hypotheses}")
                if i < len(mask_spans) -1 :
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]] for x in hypotheses]
                else:
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:] for x in hypotheses]

                # Scoring the hypotheses and select top beam hypotheses

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
                if i == len(mask_spans) -1:
                    top_beams = torch.topk(curr_loss, k=1, dim=-1, largest=False).indices
                    new_best_weighted_loss_ = curr_loss[top_beams]
                else:
                    top_beams = torch.topk(curr_loss, k=config['beam_size'], dim=-1, largest=False).indices
                
                queue = [hypotheses_all[ix] for ix in top_beams]
                
                
                # print(f"queue: {queue}")
                # for item in queue:
                #     print(item)
                # print('')
                
            elif method == "1_3":
                hypotheses=list(queue) # deletion case
                hypotheses.extend(get_beam_hypotheses_v0_variable_length_v2(source_text, 
                                        masked_sequence, 
                                        (indices_in_mlm_tokens_0, indices_in_mlm_tokens_1),
                                        predicted_token_ids.indices,
                                        mlm_tokenizer, 
                                        lossfns,
                                        config,
                                        return_all_hypotheses=True,
                                        primary_loss_only=True)[0][0])
                # print(f"hypotheses: {hypotheses}")
                if i < len(mask_spans) -1 :
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:mask_spans[i+1][0]] for x in hypotheses]
                else:
                    hypotheses_all = [x + test_sent_merged[mask_spans[i][1]:] for x in hypotheses]

                # Scoring the hypotheses and select top beam hypotheses

                curr_loss = torch.zeros(len(hypotheses_all)).to(config['device'])
                data_loader = DataLoader(CustomDataset(hypotheses_all),batch_size=batch_size)

                for lossid, lossname in enumerate(config["losses"]):
                    if (i <  len(mask_spans) -1) and (lossid > 0): # only use primary loss if step hasn't reached the end
                        break
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
                if i == len(mask_spans) -1:
                    top_beams = torch.topk(curr_loss, k=1, dim=-1, largest=False).indices
                    new_best_weighted_loss_ = curr_loss[top_beams]
                else:
                    top_beams = torch.topk(curr_loss, k=config['beam_size'], dim=-1, largest=False).indices
                
                queue = [hypotheses_all[ix] for ix in top_beams]
                
                
                # print(f"queue: {queue}")
                # for item in queue:
                #     print(item)
                # print('')
                
            torch.cuda.empty_cache()
        results.append(queue)
        
# print(time.time()-start)

with open(f'new_module/decoding_time_using_v{method}.txt', 'w') as f:
    f.write(str(time.time()-start))
    

import joblib
joblib.dump(results, f'new_module/decoding_result_using_v{method}.pkl')
