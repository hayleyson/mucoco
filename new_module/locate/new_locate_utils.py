import string
import os
import random
from typing import List, Tuple
from copy import deepcopy
from itertools import repeat
import logging

import pandas as pd
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

from new_module.em_training.nli.models import EncoderModel
torch.set_printoptions(precision=10)


logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGGING_LEVEL", logging.DEBUG))

random.seed(42)

def get_word2tok(row: pd.Series, tokenizer: AutoTokenizer) -> dict:
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
    # word2tok = dict(zip(range(len(grouped_tokens)), grouped_tokens))
    # return word2tok
    return tok2word, grouped_tokens

def get_word_level_locate_indices(current_sent:str,prediction:list,length:int, top_masks_final:list, tokenizer:AutoTokenizer, task:str):
    # word의 일부만 locate 한 경우, word 전체를 locate 한다.
    # 같은 word 안에 있는 token 끼리 묶음.
    words_raw = current_sent.strip().split()
    words = words_raw
    if task == "nli": 
        # For nli, simple spliting might not have split cases like "<s>hello" or "hello</s>world" or "world</s>"
        # So post-process to split those cases.
        # Not generalized to all possible ways a tokenizer handles premise & hypothesis
        # Currently support either one of the two 
        # 1) <s>..premise..</s>..hypothesis..</s> 
        # 2) <s>..premise..</s></s>..hypothesis..</s> 
        assert tokenizer.eos_token == tokenizer.sep_token, "Can only deal with case where EOS == SEP" ## 이런 경우만 처리할 수 있다.
        BOS=tokenizer.bos_token
        EOS=tokenizer.eos_token  
        SEP = EOS*2 if EOS*2 in current_sent else EOS
        words = []
        for w in words_raw:
            if w.strip() == "":
                continue
            if w.strip() == BOS or w.strip() == EOS or w.strip() == SEP:
                words.append(w)
                continue
            
            starts_with_BOS = w.startswith(BOS)
            ends_with_EOS = w.endswith(EOS)
            contains_SEP = SEP in w.rstrip(EOS)  # have to do w.rstrip(EOS) because it can be that SEP == EOS

            tmp_words = []
            if starts_with_BOS:
                tmp_words.append(BOS)
                w = w.lstrip(BOS)
            if contains_SEP:
                tmp_w = w.split(SEP)
                if tmp_w[0] != "":
                    tmp_words.append(tmp_w[0])
                tmp_words.append(SEP)
                w = EOS.join(tmp_w[1:])
            if ends_with_EOS:
                if w.rstrip(EOS) != "":
                    tmp_words.append(w.rstrip(EOS))
                tmp_words.append(EOS)
                w = ""
            if w != "":
                tmp_words.append(w)
                
            words.extend(tmp_words)
                
    prediction = prediction[:length]
    tok2word, grouped_tokens = get_word2tok(pd.Series({'words':words, 'tokens':prediction}), tokenizer)
    
    top_masks_final.sort()
    word_indices = []
    for index in top_masks_final:
        if index not in word_indices:
            word_index = tok2word.get(index, None)
            if word_index is not None:
                word_indices.extend(grouped_tokens[word_index])
            else:
                word_indices.append(index)    
    return list(set(word_indices))

class LocateMachine:
    def __init__(self, model, tokenizer, task):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        try:
            self.device = model.device
        except:
            self.device = model.params['device']
        
        punctuations = string.punctuation + '\n '
        punctuations = list(punctuations)
        punctuations.remove('-')
        stopwords = [" and", " of", " or", " so"] + punctuations + [token for token in self.tokenizer.special_tokens_map.values()]
        self.stopwords_ids = self.tokenizer.batch_encode_plus(stopwords, return_tensors="pt",add_special_tokens=False)['input_ids'].squeeze().to(self.device)

    def extract_hypothesis(self, text):
        text = text[:-len(self.tokenizer.eos_token)].rstrip() # remove eos_token at the end of the sentence
        text = text.split(self.tokenizer.sep_token)[-1] # take only hypothesis
        return text

    def locate_main(self, prediction, method, max_num_tokens = 7, unit="word",**kwargs):
        
        if kwargs.get('tokenized_input', False):
            batch = deepcopy(prediction)
        else:
            batch = self.tokenizer(prediction, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt").to(self.device) # prediction이 list여도 처리가능함
        
        if method == "attention":
            try:
                output = self.model(**batch, output_attentions=True)
                attentions = output.attentions
            except:
                logits, attentions = self.model(**batch)
            ## attentions : tuple of length num hidden layers
            ## attentions[i] : attention value of ith hidden layer of shape (batch, num_heads, query, value)            
            attentions = attentions[kwargs['num_layer']]
            token_wise_scores = attentions.max(1)[0][:, 0] # cls_attns's dimension: (N, L)
            
        elif method == "grad_norm":
            try:
                output = self.model(**batch, output_hidden_states=True)
                hidden_states = output['hidden_states']
                logits = output['logits']
            except:
                logits, hidden_states = self.model(**batch)
            ## hidden_states: tuple of length num_hidden_layers
            ## hidden_states[0]: (batch_size, seq_len, hidden_size)
            layer = hidden_states[0]
            layer.retain_grad()

            try:
                if self.model.params['energynet']['output_form'] != 'real_num':
                    softmax=torch.nn.Softmax(dim=-1)
                    probs = softmax(logits)[:, kwargs['label_id']]
                else:
                    probs = logits
            except:
                softmax=torch.nn.Softmax(dim=-1)
                probs = softmax(logits)[:, kwargs['label_id']]
                
            if (kwargs.get('use_energy', False)): # if take gradient of energy
                if (type(self.model) == EncoderModel):
                    if (self.model.params['energynet']['output_form'] == '3dim_vec'):
                        (-torch.log(1-probs)).sum().backward(retain_graph=True)
                    elif (self.model.params['energynet']['output_form'] == '2dim_vec'):
                        (-torch.log(probs)).sum().backward(retain_graph=True) 
                    elif (self.model.params['energynet']['output_form'] == 'real_num'):
                        (-(probs)).sum().backward(retain_graph=True) 
                else:
                    (-torch.log(probs)).sum().backward(retain_graph=True) 
            else: # if take gradient of probability
                probs.sum().backward(retain_graph=True) ## NOTE. https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments/47026836#47026836
            
            ## layer.grad : (batch_size, seq_len, hidden_size)
            norm = torch.norm(layer.grad, dim=-1)
            ## norm : (batch_size, seq_len)
            token_wise_scores = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10))
        else:
            raise
        
        
        # create a mask to exclude special tokens (incl. PAD), stop words, etc. (e.g. premise for nli task) from being located.
        exclude_mask = (batch.attention_mask == 0) | torch.isin(batch.input_ids, self.stopwords_ids)
        if self.task == "nli":
            # sentence structure after encoding : <s> ...(premise)... </s></s> ...(hypothesis)... </s> or <s> ...(premise)... </s> ...(hypothesis)... </s> 
            # mask before the first occurrence of </s> token
            premise_mask = torch.zeros_like(batch.input_ids).bool()
            indices = (batch.input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=False)
            for i in range(batch.input_ids.size(0)):
                all_occurences = indices[indices[:, 0] == i]
                if len(all_occurences) == 0:
                    logger.warning(f"no sep token found in prediction: {prediction[i]}")
                    logger.warning(prediction[i])
                    logger.warning(batch.input_ids[i])
                first_occurence = all_occurences[0, 1]
                premise_mask[i, :first_occurence] = True
            exclude_mask |= premise_mask
        
        if (self.task == "nli") and (kwargs.get('input_includes_y', False)): # deprecated: we no longer use this option
            # if input_includes_y, then tokenized_input must also be True
            # sentence structure after encoding : <s> ...(premise)... </s> ...(hypothesis)... </s> ...(label)... </s>
            # mask after the second occurrence of </s> token
            label_mask = torch.zeros_like(batch.input_ids).bool()
            indices = (batch.input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=False)
            for i in range(batch.input_ids.size(0)):
                all_occurences = indices[indices[:, 0] == i]
                if len(all_occurences) == 0:
                    logger.warning(f"no sep token found in prediction: {prediction[i]}")
                    logger.warning(prediction[i])
                    logger.warning(batch.input_ids[i])
                second_occurence = all_occurences[1, 1]
                label_mask[i, second_occurence:] = True
            exclude_mask |= label_mask            
        
        
        # fill -inf at excluded locations and take softmax
        token_wise_scores[exclude_mask] = -float("inf")
        token_wise_scores = token_wise_scores.softmax(dim=-1)
        # calculate average among non-excluded tokens
        avg_values = token_wise_scores.sum(dim=-1) / (~exclude_mask).sum(dim=-1)

        # find number of above average tokens
        # unsqueeze to allow implicit broadcasting : (N) -> (N, 1) -> (N, L)
        top_masks = (token_wise_scores >= avg_values.unsqueeze(1))
        num_above_average_tokens = top_masks.sum(dim=-1)
        
        # get top k tokens where k = min(length/3, max_num_tokens, num_above_average_tokens)
        # k is different for each example in the batch
        lengths = batch.attention_mask.sum(dim=-1)
        if self.task == 'nli': 
            # for nli task, we locate within "hypothesis". thus, length must also only include hypothesis.
            lengths = (~((batch.attention_mask == 0) | premise_mask)).sum(dim=-1)
        
        max_num_located_tokens = torch.minimum((lengths//3), torch.LongTensor([max_num_tokens]).to(self.device))
        max_num_located_tokens = torch.minimum(max_num_located_tokens, num_above_average_tokens)
        top_masks_final = [x[:max_num_located_tokens[i]] for i,x in enumerate(token_wise_scores.argsort(dim=-1,descending=True).tolist())] 
        
        if unit == "token":
            locate_ixes_all = []
            for i, locate_ixes in enumerate(top_masks_final):
                batch.input_ids[i, locate_ixes] = self.tokenizer.mask_token_id
                locate_ixes_all.append(locate_ixes)

        elif unit == "word":
            if self.task == "nli":
                # revert lenghths to include premise
                lengths = batch.attention_mask.sum(dim=-1)
                prediction = []
                for i in range(len(batch.input_ids)):
                    prediction.append(self.tokenizer.decode(batch.input_ids[i, :lengths[i]].tolist(), skip_special_tokens=False))
            locate_ixes_all = []
            for i, arguments in enumerate(zip(prediction,batch.input_ids.tolist(), lengths.tolist(), top_masks_final, repeat(self.tokenizer), repeat(self.task))):
                locate_ixes = get_word_level_locate_indices(*arguments)
                batch.input_ids[i, locate_ixes] = self.tokenizer.mask_token_id
                locate_ixes_all.append(locate_ixes)
            
        masked_sequence_text = self.tokenizer.batch_decode(
            [x[:lengths[i]] for i, x in enumerate(batch.input_ids.tolist())]
        )
        
        if self.task == "nli":
            masked_sequence_text = [self.extract_hypothesis(x) for x in masked_sequence_text]
        
        if kwargs.get('return_scores_and_indices',False):
            if self.task == "nli":
                # For NLI task, return only hypothesis part of scores and indices
                # Find hypothesis start and end indices for each example
                hypothesis_scores = []
                hypothesis_indices = []
                for i in range(batch.input_ids.size(0)):
                    # Find first occurrence of sep_token (end of premise, start of hypothesis)
                    sep_indices = (batch.input_ids[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=False)
                    if len(sep_indices) > 0:
                        hypothesis_start = sep_indices[0].item() + 1  # +1 to start after the sep token
                        # Find end of hypothesis (last sep/eos token before padding, or end of sequence)
                        # The hypothesis ends at the final eos/sep token, which should be at lengths[i] - 1 or earlier
                        # But we want to include all hypothesis tokens, so use lengths[i] (excludes padding)
                        hypothesis_end = lengths[i]
                    else:
                        # Fallback: if no sep token found, use full length
                        hypothesis_start = 0
                        hypothesis_end = lengths[i]
                    
                    # Extract hypothesis scores (only non-premise tokens)
                    hyp_scores = token_wise_scores[i, hypothesis_start:hypothesis_end].clone()
                    hypothesis_scores.append(hyp_scores)
                    
                    # Filter indices to only include those in hypothesis and adjust to be relative
                    hyp_indices = [idx - hypothesis_start for idx in locate_ixes_all[i] 
                                  if hypothesis_start <= idx < hypothesis_end]
                    hypothesis_indices.append(hyp_indices)
                
                return masked_sequence_text, hypothesis_scores, hypothesis_indices
            else:
                return masked_sequence_text, token_wise_scores, locate_ixes_all
        
        return masked_sequence_text

class LocateMachine4SCE:
    
    def __init__(self, params, energynet, task):
        self.params = params
        self.energynet = energynet
        self.tokenizer = energynet.representation_model.tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = '.'
        self.softmax = torch.nn.Softmax(dim=-1)
        self.device = self.params['device']
        self.task = task
        
        punctuations = list(string.punctuation + '\n ')
        punctuations.remove('-')
        stopwords = [" and", " of", " or", " so"] + punctuations + [token for token in self.tokenizer.special_tokens_map.values()]
        self.stopwords_ids = self.tokenizer.batch_encode_plus(stopwords, return_tensors="pt",add_special_tokens=False)['input_ids'].squeeze().to(self.params['device'])

    def _get_word2tok(self, row: pd.Series) -> dict:
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
            
            if self.tokenizer.decode(row['tokens'][jl:jr]).strip() == row['words'][k]:
                grouped_tokens.append(list(range(jl,jr)))
                for ix in range(jl,jr):
                    tok2word[ix] = k
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1

        return tok2word, grouped_tokens

    def _get_word_level_locate_indices(self, current_sent:str,prediction:list,length:int, top_masks_final:list) -> List:
        """
        # word의 일부만 locate 한 경우, word 전체를 locate 한다.
        # 같은 word 안에 있는 token 끼리 묶음.
        """
        words = current_sent.strip().split()
        prediction = prediction[:length]
        tok2word, grouped_tokens = self._get_word2tok(pd.Series({'words':words, 'tokens':prediction}))
        
        top_masks_final.sort()
        word_indices = []
        for index in top_masks_final:
            if index not in word_indices:
                word_index = tok2word.get(index, None)
                if word_index is not None:
                    word_indices.extend(grouped_tokens[word_index])
                else:
                    word_indices.append(index)    
        return list(set(word_indices))
    
    def _calculate_token_scores(self, outputs: torch.Tensor, additional_tensor: torch.Tensor) -> torch.Tensor:
        if self.params['locate']['type'] == 'gradnorm':
            return self._calculate_token_scores_by_gradnorm(outputs, additional_tensor)
        elif self.params['locate']['type'] == 'attention':
            return self._calculate_token_scores_by_attention(outputs, additional_tensor)
        else:
            raise ValueError(f"Invalid locate method: {self.params['locate']['type']}")
    
    def _calculate_token_scores_by_gradnorm(self, outputs: torch.Tensor, additional_tensor: torch.Tensor) -> torch.Tensor:

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
            probs_for_incon = self.softmax(outputs)[:, 1]
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
    
    def _calculate_token_scores_by_attention(self, outputs: torch.Tensor, additional_tensor: torch.Tensor) -> torch.Tensor:
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
  
        
    
    def _locate_tokens(self, prediction: str,  token_scores: torch.Tensor, input_tokens: torch.Tensor, exclude_mask: torch.Tensor, lengths: int, max_num_tokens: int, unit: str, kwargs: dict) -> List:
        # calculate average among non-excluded tokens
        avg_values = token_scores.sum(dim=-1) / (~exclude_mask).sum(dim=-1)

        # find number of above average tokens
        # unsqueeze to allow implicit broadcasting : (N) -> (N, 1) -> (N, L)
        top_masks = (token_scores >= avg_values.unsqueeze(1))
        num_above_average_tokens = top_masks.sum(dim=-1)
        
        max_num_located_tokens = torch.minimum((lengths//3), torch.LongTensor([max_num_tokens]).to(self.device))
        max_num_located_tokens = torch.minimum(max_num_located_tokens, num_above_average_tokens)
        top_masks_final = [x[:max_num_located_tokens[i]] for i,x in enumerate(token_scores.argsort(dim=-1,descending=True).tolist())] 
        
        if unit == "token":
            locate_ixes_all = []
            for i, locate_ixes in enumerate(top_masks_final):
                input_tokens[i, locate_ixes] = self.tokenizer.mask_token_id
                locate_ixes_all.append(locate_ixes)

        elif unit == "word":
            locate_ixes_all = []
            for i, arguments in enumerate(zip(prediction,input_tokens.tolist(), lengths.tolist(), top_masks_final)):
                locate_ixes = self._get_word_level_locate_indices(*arguments)
                input_tokens[i, locate_ixes] = self.tokenizer.mask_token_id
                locate_ixes_all.append(locate_ixes)
            
        masked_sequence_text = self.tokenizer.batch_decode(input_tokens)
        
        
        if kwargs.get('return_scores_and_indices',False):
            return masked_sequence_text, token_scores, locate_ixes_all
        
        return masked_sequence_text
            
            
    
    def _locate_instance(self, token_scores: torch.Tensor, instance_locations: List, batch_size: int) -> List:
        
        # calculate instance-level score
        instance_scores = []

        if self.params['locate']['agg_method'] == 'max':
            # calculate max token score within each instance 
            for b in range(batch_size):
                instance_scores.append([
                    token_scores[b][start:end].max().item() 
                    for start, end in instance_locations[b]
                ])
        
        elif self.params['locate']['agg_method'] == 'avg':
            # calculate average of token scores within each instance
            # for denominator, only consider nonzero values (=exclude stopwords)
            for b in range(batch_size):
                batch_scores = []
                for start, end in instance_locations[b]:
                    instance_tokens = token_scores[b][start:end]
                    nonzero_count = instance_tokens.nonzero().shape[0]
                    if nonzero_count > 0:
                        batch_scores.append(instance_tokens.sum().item() / nonzero_count)
                    else:
                        batch_scores.append(0.0)
                instance_scores.append(batch_scores)
       
        elif self.params['locate']['agg_method'] == 'median':
            # calculate median of token scores within each instance
            for b in range(batch_size):
                instance_scores.append([
                    token_scores[b][start:end].median().item() 
                    for start, end in instance_locations[b]
                ])


        # choose instances to detect
        if self.params['locate']['select_method'] in ['max', 'recursive_max']:
            thresholds = []
            prediction_list = []
            for b in range(batch_size):
                # Handle edge case where all instances were filtered out
                if len(instance_scores[b]) == 0:
                    logger.warning(f"No valid instances found for batch {b}. All instances contain only masked tokens.")
                    prediction_list.append([])
                else:
                    thresholds.append(np.max(instance_scores[b]))
                    # add tie breaking logic
                    candidates = [i for i, score in enumerate(instance_scores[b]) if score == thresholds[b]]
                    if len(candidates) > 1:
                        candidates = [random.choice(candidates)]
                    prediction_list.append(candidates)
                    # print("prediction_list:", prediction_list)
                
        return prediction_list
    
    
    def _extract_instances(self, set_text):

        # set_text == text, e.g., '<s> qa pair 1 </s> qa pair 2 ... </s>

        out = set_text[len(self.cls_token):].split(self.sep_token)[:-1]
        
        return [o+self.sep_token for o in out]
    
    
    def _detect_instance_start_end_indexes(self, string_inputs: List[str]) -> Tuple[List, List]:
        """
        Detect instances within the input text and return their start and end indexes.
        
        Args:
            string_inputs: List of input strings
            tokenized_input: Tokenized input tensor (currently unused but kept for API consistency)
            
        Returns:
            Tuple of (num_instances_per_batch, instance_locations_per_batch)
        """
        if self.energynet.decomposition_type != 'no':
            raise ValueError(f"Invalid decomposition type: {self.energynet.decomposition_type}")
        
        # Detect spans for each input string
        instances_per_batch = [self._extract_instances(string) for string in string_inputs]
        
        # Encode each instance and calculate lengths
        instance_locations_per_batch = []
        num_instances_per_batch = []
        
        for instances in instances_per_batch:
            instance_lengths = [
                len(self.tokenizer.encode(instance, add_special_tokens=False))
                for instance in instances
            ]
            
            # Calculate cumulative positions starting from 0
            # For lengths [a, b, c], cumsum gives [a, a+b, a+b+c]
            # We prepend 1 to get [1, a+1, a+b+1, a+b+c+1] (because cls token is added)
            cumulative = np.cumsum([1] + instance_lengths)
            
            # Create (start, end) pairs: (cumulative[i], cumulative[i+1])
            instance_locations = [
                (cumulative[i], cumulative[i + 1])
                for i in range(len(instances))
            ]
            
            instance_locations_per_batch.append(instance_locations)
            num_instances_per_batch.append(len(instance_locations))
        
        return num_instances_per_batch, instance_locations_per_batch
    
    def _instance_preserving_encode_plus(self, string_inputs: List[str]) -> torch.Tensor:
        
        # Detect spans for each input string
        instances_per_batch = [self._extract_instances(string) for string in string_inputs]
        # logger.debug(f"instances_per_batch: {instances_per_batch}")
        
        # Encode each instance
        instance_encoded_per_batch = []        
        for instances in instances_per_batch:
            instance_encoded = [
                self.tokenizer.encode(instance, add_special_tokens=False)
                for instance in instances
            ]
            # logger.debug(f"instance_encoded: {instance_encoded}")
            # flatten the list
            instance_encoded = sum(instance_encoded, [])
            instance_encoded = [self.tokenizer.cls_token_id] + instance_encoded
            instance_encoded_per_batch.append(torch.LongTensor(instance_encoded))
            
        input_tensor = torch.nn.utils.rnn.pad_sequence(instance_encoded_per_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_tensor = input_tensor[:, :self.tokenizer.model_max_length]
        mask = (input_tensor != self.tokenizer.pad_token_id).long()
        
        return transformers.BatchEncoding({"input_ids": input_tensor, 
                                           "attention_mask": mask},
                                          tensor_type="pt").to(self.device)

    def locate_main(self, prediction: List[str], max_num_tokens: int = 6, mode: str = "span", unit: str = "word",**kwargs) -> Tuple[List[str], List[List[int]]]:

        """
        Locate a instance (a pair) within the input set (set of pairs). 
        
        Suppose input text looks like '<s> q1 </s> a1 </s>, q1, ..., </s>'. 
        The located instance can be anywhere in q1, a2, q2, a2, ...

        Args:
        - prediction: list of strings, each string is a pair of (question, answer)
        - max_num_tokens: maximum number of tokens to mask
        - mode: indicate whether to locate at instance level or span level, "instance" or "span"
        - unit: unit of masking, "word" or "token"
        - **kwargs: additional arguments

        Returns:
        - masked_sequence_text: list of strings, each string is the masked sequence
        - prediction_list: list of lists of integers, each list is the indexes of the located instance

        """
        
        # logger.debug(f"[new_locate_utils] prediction before adding cls token: {prediction}")
        prediction = [self.tokenizer.cls_token + " " + p.lstrip(self.tokenizer.cls_token).lstrip(" ") for p in prediction]
        # logger.debug(f"[new_locate_utils] prediction after adding cls token: {prediction}")
        outputs, hidden_states_or_attentions = self.energynet.energy_model(prediction, pair_only = True)
        # Calculate token scores
        token_scores = self._calculate_token_scores(outputs, hidden_states_or_attentions)
        
        # set additional information
        inputs = self._instance_preserving_encode_plus(prediction)
        input_tensor = inputs['input_ids']
        mask = inputs['attention_mask']
        # logger.debug(f"input_tensor: {input_tensor}")
        # logger.debug(f"mask: {mask}")
        
        _, instance_locations = self._detect_instance_start_end_indexes(prediction)       
        batch_size = input_tensor.shape[0]
        assert batch_size == 1 # this code assumes batch_size = 1
            
        # initialize return variables
        prediction_list = []
        masked_sequence_text = []
        
        # Apply attention and stopwords mask. Then take softmax
        final_mask = (mask == 0) | torch.isin(input_tensor, self.stopwords_ids)
        token_scores[final_mask] = -float("inf")
        token_scores = token_scores.softmax(dim=-1)
        
        # Filter out degenerate instances (those with only masked tokens)
        # This prevents division by zero errors in instance scoring
        filtered_instance_locations = []
        filtered_instance_indexes = []
        for b in range(batch_size):
            valid_instances = []
            for j, (start, end) in enumerate(instance_locations[b]):
                # Check if instance has at least one non-masked token
                instance_has_nonmasked = (~final_mask[b][start:end]).any().item()
                if instance_has_nonmasked:
                    valid_instances.append((start, end))
                    filtered_instance_indexes.append(j)
                else:
                    logger.info(f"Filtering out degenerate instance at ({start}, {end}) with only masked tokens")
            filtered_instance_locations.append(valid_instances)
        
        # Update instance_locations to use only valid instances
        instance_locations = filtered_instance_locations
        
        # First locate at instance-level
        prediction_list = self._locate_instance(token_scores, instance_locations, batch_size)
        prediction_list_adjusted = [[filtered_instance_indexes[_idx] for _idx in prediction_list[0]]]
        
        if mode == "span":
            # Mask tokens that are not in located instances
            instance_mask = torch.zeros_like(token_scores, dtype=torch.bool)
            for b in range(batch_size):
                for instance_idx in prediction_list[b]:
                    instance_mask[b][instance_locations[b][instance_idx][0]:instance_locations[b][instance_idx][1]] = True
            instance_mask = ~instance_mask
            token_scores[instance_mask] = -float("inf")
            token_scores = token_scores.softmax(dim=-1)
            
            length_mask = (mask == 1) & ~instance_mask
            lengths = length_mask.sum(dim=-1)
            
            # Then locate & mask tokens within identified instances
            masked_sequence_text = self._locate_tokens(prediction, token_scores, input_tensor, final_mask, lengths, max_num_tokens, unit, kwargs)
            
            # logger.debug(f"masked_sequence_text before stripping cls token: {masked_sequence_text}")
            masked_sequence_text = [m[len(self.tokenizer.cls_token):].lstrip(" ") for m in masked_sequence_text]
            # logger.debug(f"masked_sequence_text after stripping cls token: {masked_sequence_text}")
            
            return masked_sequence_text, prediction_list_adjusted
        elif mode == "instance":
            # Create text with the located instance removed
            predicted_instance_start, predicted_instance_end = instance_locations[0][prediction_list[0][0]]
            new_input_tensor = torch.cat([input_tensor[:, :predicted_instance_start], input_tensor[:, predicted_instance_end:]], axis=-1) if predicted_instance_start > 0 else input_tensor[:, predicted_instance_end:]
            new_prediction = self.tokenizer.batch_decode(new_input_tensor)
            new_prediction = new_prediction[0].strip('<s>').strip(' ')

            # Extract the located instance
            predicted_instance_tensor = input_tensor[:, predicted_instance_start: predicted_instance_end]
            predicted_instance = self.tokenizer.batch_decode(predicted_instance_tensor)
            predicted_instance = predicted_instance[0].strip('<s>').strip(' ')

            return (new_prediction, predicted_instance), prediction_list_adjusted
    
    def verify_consistency(self, text):
        

        with torch.no_grad():
            # set consistency verification
            batch_text = ['<s> ' + text]
            output, _ = self.energynet.energy_model(batch_text, pair_only = True)
            
            if (self.energynet.output_form == 'real_num'):
                probs = output.reshape(-1)
            else:
                raise ValueError(f"Unsupported output form: {self.energynet.output_form}")
        
            #  classify

            cons = torch.where(probs <= self.energynet.threshold,1,0).item()
        if cons == 1:
            return 'con'
        else:
            return 'incon'

    def locate_multiple_instances_at_once(self, text):
        predicted_indexes = []
        remaining_index_list = list(range(len(self._extract_instances(text))))

        while ((self.verify_consistency(text) == 'incon') and len(text) > 0):
            
            (text,_), index_list = self.locate_main([text], mode="instance")
            index = index_list[0][0]
            
            predicted_indexes.append(remaining_index_list[index])
            remaining_index_list = (remaining_index_list[:index] if index > 0 else []) + remaining_index_list[index+1:] 
        
        return sorted(predicted_indexes)

    
    
if __name__ == "__main__":
    
    import os
    import sys
    
    import argparse
    import time
    import json
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    
    from new_module.em_training.nli.models import EncoderModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--label_id", type=int)
    parser.add_argument("--max_num_tokens", type=int, default=7)
    parser.add_argument("--locate_method", type=str, default='grad_norm')
    args = parser.parse_args()

    # 모델과 토크나이저 불러오기
    pretrained_model_path = args.pretrained_model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.task == "nli":
        # config
        with open(os.path.join(pretrained_model_path, 'config.json')) as f:
            model_config = json.load(f)
        model_config['device'] = device
        model_config['model_path'] = os.path.join(pretrained_model_path, 'best_model_pearsonr.pth')
        if args.locate_method == "attention":
            model_config['locate']['type'] = "attention"
        elif args.locate_method == "grad_norm":
            model_config['locate']['type'] = "gradnorm"
        
        # load model
        model = EncoderModel(params=model_config)
        model.load_state_dict(torch.load(model_config['model_path'],weights_only=True),strict=False)
        model.eval()
        model.to(device)
        
        tokenizer = model.tokenizer
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = model.to(device)

    # LocateMachine 초기화
    locator = LocateMachine(model, tokenizer, args.task)

    # 입력 JSONL 파일 경로
    input_file = args.input_file

    # 출력 JSONL 파일 경로
    output_file = args.output_file
    # 출력 JSONL 저장 디렉토리 생성 (이미 있으면 Skip)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 실행시간 파일 경로
    execution_time_file = output_file.replace(".jsonl", ".time")

    # print("job id:", job_id)
    print("pretrained model path:", pretrained_model_path)
    print("input file path:", input_file)
    print("output file path:", output_file)

    print("Locating Start...")
    start_time = time.time()

    # 입력 파일 열기
    with open(input_file, 'r', encoding='utf-8') as infile:
        # 출력 파일 열기
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if args.task == "formality":
                    text = line.rstrip()
                    # locate_main 적용
                    masked_text, scores, indices = locator.locate_main([text], 
                                                      args.locate_method, 
                                                      max_num_tokens=args.max_num_tokens, 
                                                      unit='word', 
                                                      label_id=args.label_id,
                                                      num_layer=10,
                                                      return_scores_and_indices=True
                                                      )
                    data = masked_text[0]
                    outfile.write(data)
                else:   
                    # JSON 형식으로 변환
                    data = json.loads(line)
                    if args.task == "toxicity_extended":
                        prompt = ""
                        generations = [data]
                    else:
                        prompt = data['prompt']['text']
                        generations = data['generations']
                    
                    # generations 내의 각 text에 대해 LocateMachine 적용
                    for generation in generations:
                        if args.task == "nli":
                            text = f"<s>{prompt}</s>{generation['text']}</s>"
                        else:
                            text = generation['text']
                        # locate_main 적용
                        masked_text, scores, indices = locator.locate_main([text], 
                                                          args.locate_method, 
                                                          max_num_tokens=args.max_num_tokens, 
                                                          unit='word', 
                                                          label_id=args.label_id,
                                                          num_layer=10,
                                                          return_scores_and_indices=True)
                        # masked 결과를 generation에 추가 (기존 key나 새로운 key 사용 가능)
                        generation['text'] = masked_text[0]  # locate_main은 리스트를 반환하므로 첫 번째 값 선택
                    
                        generation['roberta_token_pred_scores'] = [round(x, 4) for x in scores[0].tolist()]
                        generation['roberta_token_pred_indexes'] = indices[0]
                    
                    # 결과를 다시 JSON 형식으로 변환하고 출력 파일에 쓰기
                    json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

    end_time = time.time()

    # 실행 시간 계산 및 출력
    execution_time = (end_time - start_time)
    
    
    with open(execution_time_file, 'w') as f:
        f.write(str(execution_time) + "\n")
        
    print(f"Code execution time: {execution_time:.2f} seconds")