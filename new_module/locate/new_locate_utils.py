import string
from typing import List
from copy import deepcopy

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from itertools import repeat 
from new_module.em_training.nli.models import EncoderModel

# import torch.multiprocessing as mp ## not needed since not using multiprocessing
# import os ## not needed since not using multiprocessing

# os.environ['TOKENIZERS_PARALLELISM']='true' ## not needed since not using multiprocessing

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
    words = words_ = current_sent.strip().split()
    if task == "nli":
        words = []
        for w in words_:
            if ('<s>' in w):
                w = ['<s>', w.replace('<s>', '')]
            elif ('</s></s>' in w):
                w = w.split('</s></s>')
                w.insert(1, '</s></s>')
            elif ('</s>' in w):
                w = [w.replace('</s>', ''), '</s>']
            else:
                w = [w]
            words.extend(w)
                
    prediction = prediction[:length]
    tok2word, grouped_tokens = get_word2tok(pd.Series({'words':words, 'tokens':prediction}), tokenizer)
    
    top_masks_final.sort()
    top_masks_final_final = []
    for index in top_masks_final:
        if index not in top_masks_final_final:
            # word = [grouped_ixes for grouped_ixes in grouped_tokens if index in grouped_ixes]
            word_index = tok2word.get(index, None)
            # if len(word) > 0:
            if word_index is not None:
                top_masks_final_final.extend(grouped_tokens[word_index])
            else:
                top_masks_final_final.extend([index])    
    return list(set(top_masks_final_final))

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

    def locate_main(self, prediction, method, max_num_tokens = 6, unit="word",**kwargs):
        
        if kwargs.get('tokenized_input', False):
            batch = deepcopy(prediction)
        else:
            batch = self.tokenizer(prediction, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt").to(self.device) # prediction이 list여도 처리가능함
        
        if method == "attention":
            output = self.model(**batch, output_attentions=True)
            attentions = output.attentions
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
                    print(prediction[i])
                    print(batch.input_ids[i])
                first_occurence = all_occurences[0, 1]
                premise_mask[i, :first_occurence] = True
            exclude_mask |= premise_mask
        
        if (self.task == "nli") and (kwargs.get('input_includes_y', False)):
            # if input_includes_y, then tokenized_input must also be True
            # sentence structure after encoding : <s> ...(premise)... </s> ...(hypothesis)... </s> ...(label)... </s>
            # mask after the second occurrence of </s> token
            label_mask = torch.zeros_like(batch.input_ids).bool()
            indices = (batch.input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=False)
            for i in range(batch.input_ids.size(0)):
                all_occurences = indices[indices[:, 0] == i]
                if len(all_occurences) == 0:
                    print(prediction[i])
                    print(batch.input_ids[i])
                second_occurence = all_occurences[1, 1]
                label_mask[i, second_occurence:] = True
            exclude_mask |= label_mask            
        
        
        # fill -inf at excluded locations and take softmax
        token_wise_scores[exclude_mask] = -float("inf")
        token_wise_scores = token_wise_scores.softmax(dim=-1)
        # calculate average among non-excluded tokens
        avg_values=token_wise_scores.sum(dim=-1)/(~exclude_mask).sum(dim=-1) # tensor([0.5120, 0.3744], device='cuda:0', grad_fn=<DivBackward0>)

        # find number of above average tokens
        top_masks = (token_wise_scores >= avg_values.unsqueeze(1))# unsqueeze to allow implicit broadcasting : (N) -> (N, 1) -> (N, L)
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
        ## clean up special tokens other than <mask> token for nli task
        if self.task == "nli":
            masked_sequence_text = [x.strip(self.tokenizer.eos_token).split(self.tokenizer.sep_token)[-1] for i, x in enumerate(masked_sequence_text)]
        
        if kwargs.get('return_scores_and_indices',False):
            return masked_sequence_text, token_wise_scores, locate_ixes_all
        return masked_sequence_text
    
if __name__ == "__main__":
    
    import os
    import sys
    os.chdir('/data/hyeryung/mucoco')
    sys.path.append(os.path.abspath('.'))

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
                    masked_text = locator.locate_main([text], 
                                                      'grad_norm', 
                                                      max_num_tokens=args.max_num_tokens, 
                                                      unit='word', 
                                                      label_id=args.label_id)
                    data = masked_text[0]
                    outfile.write(data)
                else:   
                    # JSON 형식으로 변환
                    data = json.loads(line)
                    prompt = data['prompt']['text']
                    generations = data['generations']
                    
                    # generations 내의 각 text에 대해 LocateMachine 적용
                    for generation in generations:
                        text = f"<s>{prompt}</s>{generation['text']}</s>" if args.task == "nli" else generation['text']
                        # locate_main 적용
                        masked_text = locator.locate_main([text], 
                                                          'grad_norm', 
                                                          max_num_tokens=args.max_num_tokens, 
                                                          unit='word', 
                                                          label_id=args.label_id)
                        # masked 결과를 generation에 추가 (기존 key나 새로운 key 사용 가능)
                        generation['text'] = masked_text[0]  # locate_main은 리스트를 반환하므로 첫 번째 값 선택
                    
                    
                    # 결과를 다시 JSON 형식으로 변환하고 출력 파일에 쓰기
                    json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

    end_time = time.time()

    # 실행 시간 계산 및 출력
    execution_time = (end_time - start_time) / 60
    print(f"Code execution time: {execution_time:.2f} minutes")