"""
editing_with_delete_variable_replace 함수 업데이트를 위한 프로토타입,테스트 코드
(v0 와 v1을 비교하면서, 여러 fluency_em에 따른 차이도 비교)
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

from new_module.algo_improve.new_decode_utils_old_debug import get_beam_hypotheses_v0, get_beam_hypotheses_v1, get_combi_hypotheses, final_reranking, analyze_span_lengths_and_count, editing_with_delete_variable_replace
from new_module.algo_improve.new_decode_utils_new_debug import editing_with_delete_variable_replace as editing_with_delete_variable_replace_new

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



if __name__ == "__main__":
    
    
    # 각 방안 별로 결과가 어떻게 달라지는지 보다 여러 샘플에서 확인하기 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--fluency_em_path", type=str, default='Qwen/Qwen2.5-7B')
    args = parser.parse_args()
    
    # prototype위한 변수 세팅
    config = {'task': 'toxicity',
        'device': 'cuda',
        'losses': ['gpt2', 'classification_no_prefix_logprobloss'],
        'cache_dir': '/data/hyeryung/hf_cache',
        'model_paths': [args.fluency_em_path,
                        '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint'],
        'model_types': ['AutoModelForCausalLM', 'AutoModelForSequenceClassification'],
        'build_loss_dict': {'AR_top_k': 0,
                            'AR_top_p': 0.96,
                            'loss_type': 'xentropy',
                            'coeff_steps': 200,
                            'coeff_pattern': 'constant',
                            'AR_temperature': 1,
                            'length_normalize': False,
                            'max_output_length': 20},
        'tokenizer_paths': [args.fluency_em_path,
                        '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint'],
        'max_tokens_per_span': 3,
        'consider_prompt_for_cand_gen': True,
        'k_per_location': 10,
        'loss_weights': [0.1, 1.0],
        'target_label_ids': [0, 0],
        'beam_size': 5,
        'min_epsilons': [0.95],
        'selection_criteria': 'allsat_primary'
    }
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

    name2model[config["model_paths"][0]].half() ## half to speed up experiments

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
    
    with open('/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332_index.txt', 'r') as f:
        indices = f.read().split()
    indices = [int(x) for x in indices]

    located_data = pd.read_json('/data/hyeryung/mucoco/new_module/locate/locate_num_tokens_eda/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_nontoxic_locate_max_7.jsonl', lines=True)
    located_data = located_data.explode('generations').reset_index(drop=True)
    located_data = located_data.loc[located_data['generations'].apply(len) != 0].reset_index(drop=True)

    located_data['prompt'] = located_data['prompt'].apply(lambda x: x['text'])
    located_data['masked_sentences'] = located_data['generations'].apply(lambda x: x['text'])
    located_data = located_data.loc[indices,:].copy()
    print(f"Number of total samples: {len(located_data)}")
    all_source_texts = located_data['prompt'].tolist()
    all_masked_sentences = located_data['masked_sentences'].tolist()
    
    mask_info_dicts = []
    span_lengths_es = []

    for test_sent in all_masked_sentences:
        
        mask_info_dict, span_lengths = analyze_span_lengths_and_count(test_sent)
        mask_info_dicts.append(mask_info_dict)
        span_lengths_es.append(span_lengths)
        
        

    torch.cuda.empty_cache()
    method = args.method
    start = time.time()
    results = []


    # 1) span 돌면서 decoding
    # 2) final로는 1개의 candidate만 나옴 
    ## editing_with_delete_variable_replace
    ## return 되어야 할 값들 목록 : final_hypotheses, new_best_weighted_loss_, new_best_allsat_, new_best_logging_loss_

    # random.seed(999)
    with open('/data/hyeryung/mucoco/new_module/_notebooks/nb_debug_le_algorithm_update_20241215_indexes.txt', 'r') as f:
        idxes_for_test = f.read().split()
    idxes_for_test = [int(x) for x in indices]
    # idxes_for_test = random.sample(range(len(all_source_texts)),args.num_test_samples)
    # print(f"Number of samples: {len(idxes_for_test)}")

    batch_size = 64
    loss_weights = config['loss_weights']

    for sample_idx in tqdm.tqdm(idxes_for_test[:]):
        
        source_text = all_source_texts[sample_idx]
        test_sent = all_masked_sentences[sample_idx]
        test_sent_span_lengths = span_lengths_es[sample_idx]
        print(f"source_text: {source_text}")
        print(f"test_sent: {test_sent}")

        if method == "0":
            final_hypotheses_curr, new_best_weighted_loss_curr, new_best_allsat_curr, new_best_logging_loss_curr = \
                    editing_with_delete_variable_replace(source_text, test_sent, test_sent_span_lengths, mlm, mlm_tokenizer, lossfns, config)
            results.append(final_hypotheses_curr)
        elif method == "1":
            final_hypotheses_curr, new_best_weighted_loss_curr, new_best_allsat_curr, new_best_logging_loss_curr = \
                    editing_with_delete_variable_replace_new(source_text, test_sent, test_sent_span_lengths, mlm, mlm_tokenizer, lossfns, config)
            results.append(final_hypotheses_curr)
            
    # print(time.time()-start)
    execution_time = time.time()-start
    with open(f"new_module/decoding_time_using_v{method}_{args.fluency_em_path.split('/')[-1]}.txt", 'w') as f:
        f.write(str(execution_time))
    
    import joblib
    joblib.dump(results, f"new_module/decoding_result_using_v{method}_{args.fluency_em_path.split('/')[-1]}.pkl")

    
    del lossfns, name2config, name2model, name2tokenizer, mlm, mlm_tokenizer
    torch.cuda.empty_cache()
    
    