import string
from typing import List

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from itertools import repeat 
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

def get_word_level_locate_indices(current_sent:str,prediction:list,length:int, top_masks_final:list, tokenizer:AutoTokenizer):
    # word의 일부만 locate 한 경우, word 전체를 locate 한다.
    # 같은 word 안에 있는 token 끼리 묶음.
    words = current_sent.strip().split()
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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        punctuations = string.punctuation + '\n '
        punctuations = list(punctuations)
        punctuations.remove('-')
        stopwords = [" and", " of", " or", " so"] + punctuations + [token for token in self.tokenizer.special_tokens_map.values()]
        self.stopwords_ids = self.tokenizer.batch_encode_plus(stopwords, return_tensors="pt",add_special_tokens=False)['input_ids'].squeeze().to(self.model.device)

    def locate_main(self, prediction: List[str], method, max_num_tokens = 6, unit="word",**kwargs):
        
        if self.args.task == "nli":
            premises, hypotheses = list(map(list, zip(*prediction)))
            batch = self.tokenizer.batch_encode_plus(premises, hypotheses, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True).to(self.device)
        else:
            batch = self.tokenizer(prediction, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt").to(self.model.device) # prediction이 list여도 처리가능함
        lengths = batch.attention_mask.sum(dim=-1)
        
        if method == "attention":
            output = self.model(**batch, output_attentions=True)
            attentions = output.attentions
            ## attentions : tuple of length num hidden layers
            ## attentions[i] : attention value of ith hidden layer of shape (batch, num_heads, query, value)            
            attentions = attentions[kwargs['num_layer']]
            token_wise_scores = attentions.max(1)[0][:, 0] # cls_attns's dimension: (N, L)
            
        elif method == "grad_norm":
            output = self.model(**batch, output_hidden_states=True)
            ## output['hidden_states']: tuple of length num_hidden_layers
            ## output['hidden_states'][0]: (batch_size, seq_len, hidden_size)
            layer = output['hidden_states'][0]
            layer.retain_grad()

            softmax=torch.nn.Softmax(dim=-1)
            probs = softmax(output['logits'])[:, kwargs['label_id']]
            probs.sum().backward(retain_graph=True) ## NOTE. https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments/47026836#47026836
            ## layer.grad : (batch_size, seq_len, hidden_size)
            norm = torch.norm(layer.grad, dim=-1)
            ## norm : (batch_size, seq_len)
            token_wise_scores = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10))
        else:
            raise
        
        ## avg_value만 구하면 된다고 한다면, 이렇게도 가능하다.
        ## softmax 먼저 해주기
        token_wise_scores[batch.attention_mask == 0] = -float("inf")
        token_wise_scores = token_wise_scores.softmax(dim=-1)
        
        token_wise_scores[torch.isin(batch.input_ids, self.stopwords_ids)]=0.0
        no_punc_len=(~torch.isin(batch.input_ids, self.stopwords_ids)).sum(dim=-1) # tensor([2, 4], device='cuda:0')
        # no_punc_len=(~torch.isin(batch.input_ids, self.stopwords_ids)*batch.attention_mask).sum(dim=-1) # tensor([2, 4], device='cuda:0')
        avg_values=token_wise_scores.sum(dim=-1)/no_punc_len # tensor([0.5120, 0.3744], device='cuda:0', grad_fn=<DivBackward0>)

        ## stopwords가 아닌 부분에서만 index를 찾아야 한다.
        token_wise_scores[torch.isin(batch.input_ids, self.stopwords_ids)]=-float("inf") ## stopwords 인부분을 -inf로 세팅하면, avg랑 비교할 때랑 topk를 찾을때 빠질것으로 예상
        top_masks = (token_wise_scores >= avg_values.unsqueeze(1))# unsqueeze to allow implicit broadcasting : (N) -> (N, 1) -> (N, L)
        max_num_located_tokens = torch.minimum((lengths//3), torch.LongTensor([max_num_tokens]).to(self.model.device))
        max_num_located_tokens = torch.minimum(max_num_located_tokens, top_masks.sum(dim=-1))
        top_masks_final = [x[:max_num_located_tokens[i]] for i,x in enumerate(token_wise_scores.argsort(dim=-1,descending=True).tolist())] 

        if unit == "token":
            for i, locate_ixes in enumerate(top_masks_final):
                batch.input_ids[i, locate_ixes] = self.tokenizer.mask_token_id

        elif unit == "word":

            for i, arguments in enumerate(zip(prediction,batch.input_ids.tolist(), lengths.tolist(), top_masks_final, repeat(self.tokenizer))):
                locate_ixes = get_word_level_locate_indices(*arguments)
                batch.input_ids[i, locate_ixes] = self.tokenizer.mask_token_id
                
            ## it took longer to run multiprocessing 30+ ms v.s. 8 s
            # try:
            #     mp.set_start_method('spawn', force=True)
            # except RuntimeError:
            #     pass

            # with mp.Pool(kwargs.get("num_processes", mp.cpu_count())) as pool:
            #     # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
            #     locate_ixes = pool.starmap(get_word_level_locate_indices, zip(prediction,batch.input_ids.tolist(), lengths.tolist(), top_masks_final, repeat(self.tokenizer)))
        
        masked_sequence_text = self.tokenizer.batch_decode(
            [x[:lengths[i]] for i, x in enumerate(batch.input_ids.tolist())]
        )
        return masked_sequence_text
    
if __name__ == "__main__":
    
    prompt='abc'
    prediction=['dsxe<s>','sdvbfe','dsxe<s>','sdvbfe','dsxe<s>','sdvbfe','dsxe<s>','sdvbfe','dsxe<s>','sdvbfe']
    ckpt_path = '/data/hyeryung/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint/'
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    device='cuda'
    model = model.to(device)
    
    loc_machine=LocateMachine(model,tokenizer)
    res = loc_machine.locate_main(prediction, "attention", max_num_tokens = 6, unit="word", num_layer=10, label_id=0)
    print(res)