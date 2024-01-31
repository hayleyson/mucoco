import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.chdir('/data/hyeryung/mucoco')
sys.path.append("/data/hyeryung/mucoco")
import string
import torch
import pandas as pd
from transformers import XLMRobertaTokenizerFast
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""))

from new_module.locate.comet_custom import download_model, load_from_checkpoint

punctuations = string.punctuation + '\n '
punctuations = list(punctuations)
punctuations.remove('-')

def locate(attentions, tokenizer, batch, max_num_tokens = 6, num_layer=10, unit="word", use_cuda=True):

    ## attentions : tuple of length num hidden layers
    ## attentions[i] : attention value of ith hidden layer of shape (batch, num_heads, query, value)
    lengths = [i.tolist().count(1) for i in batch["attention_mask"]]
    print(lengths[1])
    # 보고자 하는 attention layer 만 가져옴
    attentions = attentions[
        num_layer # originally 10
    ]
    print(attentions.shape)
    print(attentions.max(1)[0].shape)
    print( batch["input_ids"].shape)
    print( batch["attention_mask"][0,:])
    cls_attns = attentions.max(1)[0][:, 0]
    
    stopwords = [" and", " of", " or", " so"] + punctuations + [token for token in tokenizer.special_tokens_map.values()]
    stopwords_ids = [tokenizer.encode(word,add_special_tokens=False)[-1] for word in stopwords]
    # print("stopwords_ids", torch.tensor(stopwords_ids))

    locate_ixes=[]
    for i, attn in enumerate(cls_attns):
        
        print(f"attn.shape", attn.shape)
        current_sent = batch["input_ids"][i][: lengths[i]]
        print("current_sent", current_sent)
        if use_cuda:
            no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids).to(torch.device('cuda'))))[0]
        else:
            no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids)))[0]
        print("no_punc_indices", no_punc_indices)
        print(f"current_sent[no_punc_indices]: {current_sent[no_punc_indices]}")
        print(f"tokenizer.decode(current_sent[no_punc_indices]): {tokenizer.decode(current_sent[no_punc_indices])}")
        
        # current tokenizer does not add <s> and </s> to the sentence.
        current_attn = attn[: lengths[i]].softmax(-1) 
        # print("current_attn", current_attn)
        current_attn = current_attn[no_punc_indices]
        # print("current_attn", current_attn)
        
        # 이 값의 평균을 구한다.
        avg_value = current_attn.view(-1).mean().item()
        # print("avg_value", avg_value)
        # 이 값 중에 평균보다 큰 값을 지니는 위치를 찾는다.
        # fixed to reflect that sometimes the sequence length is 1.
        top_masks = ((current_attn >= avg_value).nonzero().view(-1)) 
        torch.cuda.empty_cache()
        top_masks = top_masks.cpu().tolist()
        print("top_masks", top_masks)
        
        # attention 값이 평균보다 큰 토큰의 수가 k개 또는 문장 전체 토큰 수의 1/3 보다 크면  
        if len(top_masks) > min((lengths[i]) // 3, max_num_tokens):
            # 그냥 attention 값 기준 k 개 또는 토큰 수/3 중 작은 수를 뽑는다.
            top_masks = (
                current_attn.topk(max(min((lengths[i]) // 3, max_num_tokens), 1))[1]
            )
            top_masks = top_masks.cpu().tolist()
            # print("top k top_masks", top_masks)
        top_masks_final = no_punc_indices[top_masks]
        # print("top_masks_final", top_masks_final)
        if unit == "token":
            locate_ixes.append(list(set(top_masks_final.cpu().detach().tolist())))
        
        elif unit == "word":
            # word의 일부만 locate 한 경우, word 전체를 locate 한다.
            # 같은 word 안에 있는 token 끼리 묶음.
            j, k = 0, 0
            grouped_tokens = []
            grouped_tokens_for_word = []
            words = tokenizer.decode(current_sent).strip().split()
            # print("words", words)
            while j < len(current_sent):
                if (tokenizer.decode(current_sent[j]).strip() not in stopwords):
                    # print("tokenizer.decode(current_sent[j])", tokenizer.decode(current_sent[j]))
                    while k < len(words):
                        if tokenizer.decode(current_sent[j]).strip() in words[k]:
                            grouped_tokens_for_word.append(j)
                            break
                        else:
                            grouped_tokens.append(grouped_tokens_for_word)
                            grouped_tokens_for_word = []
                            k += 1
                j += 1
            grouped_tokens.append(grouped_tokens_for_word)
            # print("grouped_tokens", grouped_tokens)
            
            top_masks_final.sort()
            top_masks_final_final = []
            for index in top_masks_final:
                # print("index", index)
                if index not in top_masks_final_final:
                    word = [grouped_ixes for grouped_ixes in grouped_tokens if index in grouped_ixes]
                    # print("word", word)
                    if len(word) > 0:
                        word = word[0]
                    else:
                        print(f"!!! {index} not in the grouped_ixes {grouped_tokens}")
                        print(f"!!! tokenizer.decode(index): {tokenizer.decode(index)}")
                    top_masks_final_final.extend(word)
            locate_ixes.append(list(set(top_masks_final_final)))

            
    return locate_ixes


# def locate_mqm(input_path, output_path, model, tokenizer, batch_size):
def locate_mqm(input_path, output_path, tokenizer, batch_size):
    
    ## data = pd.read_csv(input_path, sep='\t', quoting=3)
    ## data['source_orig']=data['source'].str.replace('<v>', '').str.replace('</v>','')
    ## data['target_orig']=data['target'].str.replace('<v>', '').str.replace('</v>','')
    ## data_texts=data[['system','doc_id','seg_id','source_orig','target_orig']].drop_duplicates(subset=['source_orig','target_orig']); data_texts.shape
    # torch.cuda.empty_cache()
    data = pd.read_json(input_path, lines=True)
    data_texts = data.rename(columns={'source':'src', 'target_clean':'mt'})
    # data_comet=data_texts[['src','mt']].to_dict(orient='records')
    # model_output = model.predict(data_comet, batch_size=batch_size, gpus=1)
    # torch.cuda.empty_cache()
    # batches=model.prepare_sample(data_comet,stage='predict')
    # attentions_es=model_output.attentions
    # torch.save({"batches": batches, "attentions": attentions_es},
    #            f"{os.path.splitext(output_path)[0]}_batches_attentions.pt")
    
    batches_attentions=torch.load(f"{os.path.splitext(output_path)[0]}_batches_attentions.pt")
    batches=batches_attentions["batches"]
    attentions_es = batches_attentions["attentions"]
    del batches_attentions

    located_indices_all=[]
    for i in range(len(batches)):
        batch = batches[i]
        attentions = attentions_es[i]
        indices=locate(attentions, tokenizer, batch, num_layer=35, unit="word", use_cuda=False)
        located_indices_all.extend(indices)
        
    data_texts.loc[:, 'located_indices'] = located_indices_all
    data_texts.to_json(output_path,lines=True,orient='records')

if __name__ == "__main__":
    
    batch_size=32
    # model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl",saving_directory='/data/hyeryung/')
    # model = load_from_checkpoint(model_path)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("facebook/xlm-roberta-xl")

    input_path = '/data/hyeryung/loc_edit/data/wmt-mqm-2020-agg/mqm_newstest2020_ende_with_severe_error.jsonl'
    output_path = '/data/hyeryung/loc_edit/results/wmt-mqm-2020-agg/mqm_newstest2020_ende_with_severe_error_locate_attn.jsonl'
    # locate_mqm(input_path, output_path, model, tokenizer, batch_size)
    locate_mqm(input_path, output_path, tokenizer, batch_size)
    
    # input_path = '/data/hyeryung/loc_edit/data/wmt-mqm-2020-agg/mqm_newstest2020_zhen.jsonl'
    # output_path = '/data/hyeryung/loc_edit/results/wmt-mqm-2020-agg/mqm_newstest2020_zhen_locate_attn.jsonl'
    # locate_mqm(input_path, output_path, model, tokenizer, batch_size)