import random
import re
import string
from typing import List, Tuple
from itertools import repeat 

import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
torch.set_printoptions(precision=10)

class locate_by_token_scores():
    
    def __init__(self, params, energynet):
        self.params = params
        self.energynet = energynet
        self.tokenizer = energynet.representation_model.tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = '.'
        self.softmax = torch.nn.Softmax(dim=-1)
        self.params['locate']['locate_unit'] = 'span'
        
        punctuations = list(string.punctuation + '\n ')
        punctuations.remove('-')
        stopwords = [" and", " of", " or", " so"] + punctuations + [token for token in self.tokenizer.special_tokens_map.values()]
        self.stopwords_ids = self.tokenizer.batch_encode_plus(stopwords, return_tensors="pt",add_special_tokens=False)['input_ids'].squeeze().to(self.params['device'])

    def get_word2tok(self, row: pd.Series, tokenizer: AutoTokenizer) -> dict:
        """
        A function that take a list of words and a corresponding list of tokens 
        into a mapping between each word's index and its corresponding token indexes.
        @param row: A row from dataframe
        @return word2char: A dictionary with word's location index as keys and tuples of corresponding token location indexes as values.

        Example:
        row=pd.Series()
        row['words']=['wearing', 'games', 'and', 'holy', '****ing', 'shit', 'do', 'I', 'hate', 'horse', 'wearing', 'games.']
        row['tokens']=[86, 6648, 1830, 290, 11386, 25998, 278, 7510, 466, 314, 5465, 8223, 5762, 1830, 13]
        word2tok=get_word2tok(row)
        word2tok
        {0: [0, 1],
        1: [2],
        2: [3],
        ...
        10: [12],
        11: [13, 14]}
        """
        
        jl, jr, k = 0, 0, 0
        grouped_tokens = []
        tok2word=dict()
        while jr <= len(row['tokens'])+1 and k < len(row['words']):
            
            if tokenizer.decode(row['tokens'][jl:jr]).strip() == row['words'][k]:
                grouped_tokens.append(list(range(jl,jr)))
                for ix in range(jl,jr):
                    tok2word[ix] = k
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1

        return tok2word, grouped_tokens

    def get_word_level_locate_indices(self, current_sent:str,prediction:list,length:int, top_masks_final:list, tokenizer:AutoTokenizer) -> List:
        """
        # word의 일부만 locate 한 경우, word 전체를 locate 한다.
        # 같은 word 안에 있는 token 끼리 묶음.
        """
        words = current_sent.strip().split()
        prediction = prediction[:length]
        tok2word, grouped_tokens = self.get_word2tok(pd.Series({'words':words, 'tokens':prediction}), tokenizer)
        
        top_masks_final.sort()
        top_masks_final_final = []
        for index in top_masks_final:
            if index not in top_masks_final_final:
                word_index = tok2word.get(index, None)
                if word_index is not None:
                    top_masks_final_final.extend(grouped_tokens[word_index])
                else:
                    top_masks_final_final.extend([index])    
        return list(set(top_masks_final_final))
    
    def calculate_token_scores(self, outputs: torch.Tensor, additional_tensor: torch.Tensor) -> torch.Tensor:
        pass

    def locate(self, inputs: dict, outputs: torch.Tensor, additional_tensor: torch.Tensor, **kwargs) -> dict:
        
        """
        Locate a span (a pair) within the input set (set of pairs). 
        
        Suppose input text looks like '<s> q1 </s> a1 </s>, q1, ..., </s>'. 
        The located span can be anywhere in q1, a2, q2, a2, ...
        """
    
        if len(inputs)==2:
            input_tensor, mask = inputs['input_ids'], inputs['attention_mask']
            input_tensor, mask = torch.tensor(input_tensor).to(self.params['device']), torch.tensor(mask).to(self.params['device'])
        else:
            input_tensor = inputs
            mask = torch.ones_like(input_tensor)
        
        # set additional information
        num_spans, span_locations = self.detect_span(input_tensor)       
        lengths = mask.sum(dim=-1)
        batch_size = input_tensor.shape[0]
        assert batch_size == 1 # this code assumes batch_size = 1
            
        # initialize return variables
        gradient_norm_list = []
        prediction_list = []
        masked_sequence_text = []
        
        
        token_scores = self.calculate_token_scores(outputs, additional_tensor)
        
        # apply attention mask and stopwords mask
        final_mask = (mask == 0) | torch.isin(input_tensor, self.stopwords_ids)
        token_scores[final_mask] = -float("inf")
        # print("token_scores", token_scores)
        
        # take softmax
        # token_scores dimension: (batch_size, seq_len)
        token_scores = token_scores.softmax(dim=-1)
        # print("token_scores:", token_scores)
        # print("span_locations:", span_locations)
        
        # calculate span-level score
        span_scores = []
        if 'above_avg' in self.params['locate']['agg_method']:
            
            # calculate average gradient norm within each input
            nonstopword_counts = (token_scores != 0.0).sum(dim=-1) # won't include stopwords & masked tokens in average denominator
            avg_values = token_scores.sum(dim=-1) / nonstopword_counts   
            
            # find index of tokens that have above average gradient norm
            top_masks = torch.where((token_scores >= avg_values.unsqueeze(1)))[1] # unsqueeze to allow implicit broadcasting : (N) -> (N, 1) -> (N, L)
            # print("top_masks:", top_masks)
            
            # count number of above average tokens within each span
            if 'count' in self.params['locate']['agg_method']:
                # TODO: change below code if want to treat batch_size > 1
                span_scores.append([((top_masks >= l[0]) & (top_masks < l[1])).sum().item() for l in span_locations[0]])
            # calculate portion of above avereage tokens within each span 
            elif 'prop' in self.params['locate']['agg_method']:
                # TODO: change below code if want to treat batch_size > 1
                span_scores.append([((top_masks >= l[0]) & (top_masks < l[1])).sum().item() / (l[1] - l[0]) for l in span_locations[0]])
        
        elif 'max' == self.params['locate']['agg_method']:
            
            # calculate max token score within each span 
            for b in range(batch_size):
                span_scores.append([token_scores[b][l[0]:l[1]].max().item() for l in span_locations[b]])
        
        elif 'avg' == self.params['locate']['agg_method']:
            
            # calculate average of token scores within each span
            # for denominator, only consider nonzero values (=exclude stopwords)
            for b in range(batch_size):
                span_scores.append([token_scores[b][l[0]:l[1]].sum().item()/token_scores[b][l[0]:l[1]].nonzero().sum().item() for l in span_locations[b]])
       
        elif 'median' == self.params['locate']['agg_method']:
            
            # calculate average of token scores within each span
            # for denominator, only consider nonzero values (=exclude stopwords)
            for b in range(batch_size):
                span_scores.append([token_scores[b][l[0]:l[1]].median().item() for l in span_locations[b]])
        
        
        # print("span_scores:", span_scores)
        # choose spans to detect
        if self.params['locate']['select_method'] in ['above_avg', 'recursive_above_avg']:
            thresholds = []
            prediction_list = []
            for b in range(batch_size):
                thresholds.append(sum(span_scores[b]) / len(span_scores[b]))
                # print("span average:", thresholds[b])
                prediction_list.append([i for i, score in enumerate(span_scores[b]) if score >= thresholds[b]])
        
        if 'above_75p' == self.params['locate']['select_method']:
            thresholds = []
            prediction_list = []
            for b in range(batch_size):
                thresholds.append(np.percentile(span_scores[b],75))
                # print("span 75th percentile:", thresholds[b])
                prediction_list.append([i for i, score in enumerate(span_scores[b]) if score >= thresholds[b]])

        if 'above_median' == self.params['locate']['select_method']:
            thresholds = []
            prediction_list = []
            for b in range(batch_size):
                thresholds.append(np.median(span_scores[b]))
                # print("span median:", thresholds[b])
                prediction_list.append([i for i, score in enumerate(span_scores[b]) if score >= thresholds[b]])
            
        
        if self.params['locate']['select_method'] in ['max', 'recursive_max']:
            thresholds = []
            prediction_list = []
            for b in range(batch_size):
                thresholds.append(np.max(span_scores[b]))
                # add tie breaking logic
                candidates = [i for i, score in enumerate(span_scores[b]) if score == thresholds[b]]
                if len(candidates) > 1:
                    candidates = [random.choice(candidates)]
                prediction_list.append(candidates)
                # print("prediction_list:", prediction_list)
        

            
        return {
                "prediction_list": prediction_list,
                "token_scores_list": token_scores.tolist(), # gn calculation done!
                "instance_scores_list": span_scores,
                "thresholds": thresholds,
                # "errors_gold": errors
                }
    
    
    def detect_span(self, tokenized_input: torch.Tensor) -> Tuple[List, List]:
        
        if self.energynet.decomposition_type == 'no':
            # set == text, e.g., '<s> q1 </s> a1 </s>, q2, ..., </s> a2 ... </s>'
            cls_token_id = self.tokenizer.encode(self.cls_token, add_special_tokens=False)[0]
            sep_token_id = self.tokenizer.encode(self.sep_token, add_special_tokens=False)[0]
            batch_size = tokenized_input.shape[0]
            
            pair_startpoints_all = torch.where((tokenized_input == cls_token_id) | (tokenized_input == sep_token_id))
            pairs_location_list = []
            pairs_num_list = []
            for b in range(batch_size):
                
                pair_startpoints = pair_startpoints_all[1][pair_startpoints_all[0] == b].tolist()
                pairs_location = [(pair_startpoints[i]+1, pair_startpoints[i+1]+1) for i in range(len(pair_startpoints)-1)]
                pairs_location_list.append(pairs_location)
                pairs_num_list.append(len(pairs_location))
                
                # for i, (start, end) in enumerate(pairs_location):
                    # print("%d: %s" %(i, self.tokenizer.decode(tokenized_input[b][start:end])))
                
                
        return pairs_num_list, pairs_location_list
    

class locate_by_gradnorm(locate_by_token_scores):
    
    
    def calculate_token_scores(self, outputs: torch.Tensor, additional_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inputs
        @outputs: main outputs (energy score or 2-dim vector of logits) returned by an energy model
        @additional_tensor: hidden_states returned by an energy model
        
        Returns
        @token_scores: gradient norm calculated for each token in the sequence. Shape: (batch_size, sequence_length)
        """
        
        # calculate gradient norm
        additional_tensor = additional_tensor[0] # take embedding layer
        additional_tensor.retain_grad()
        if self.energynet.output_form == 'real_num':
            e_val = outputs 
            e_val.sum().backward()
        elif self.energynet.output_form == '2dim_vec':
            probs_for_incon = self.softmax1(outputs)[:, 1]
            probs_for_incon.backward()
        
        # additional_tensor.grad dimension: (batch_size, seq_len, hidden_size) 
        # => norm dimension: (batch_size, seq_len)
        # print("additional_tensor.grad", additional_tensor.grad)
        norm = torch.norm(additional_tensor.grad, dim=-1) 
        # print("norm", norm)
            
        # input a small number if there's 0.
        token_scores = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10)) 
        # print("token_scores", token_scores)
        
        return token_scores
        
    
class locate_by_attention(locate_by_token_scores):
    
    def calculate_token_scores(self, outputs, additional_tensor):
        """
        Inputs
        @outputs: main outputs (energy score or 2-dim vector of logits) returned by an energy model
        @additional_tensor: attention scores () returned by an energy model
        
        Returns
        @token_scores: attention-based scores calculated for each token in the sequence. Shape: (batch_size, sequence_length)
        """
        
        attentions = additional_tensor[self.params['locate']['attentions_num_layer']]
        attentions = attentions[:, 0] # attention weights between cls token (query) and all tokens (key)
        attentions = attentions.max(-1)[0] # max attention weight between cls token (query) and all tokens (key) calculated across multi-heads 
        token_scores = attentions
        
        return token_scores
        
    