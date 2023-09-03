import torch
import string

punctuations = string.punctuation + '\n '
punctuations = list(punctuations)
punctuations.remove('-')


# model 의 forward 함수에서 정의를 output_attentions=True를 넘길 수 있게 되어 있다.
def locate(model, tokenizer, batch, max_num_tokens = 6, num_layer=10, unit="word"):
    # torch.cuda.empty_cache()
    # forward
    model.eval()
    with torch.no_grad():
        classifier_output = model.forward(**batch, output_attentions=True)
        torch.cuda.empty_cache()
        
    # get attentions
    attentions = classifier_output["attentions"]
    # attention_mask에서 1의 개수를 셈
    # lengths = [i.tolist().count(1) for i in batch["attention_mask"]]
    lengths = [len(x) for x in batch["input_ids"]]
    # 보고자 하는 attention layer 만 가져옴
    attentions = attentions[
        num_layer # originally 10
    ]
    cls_attns = attentions.max(1)[0][:, 0]
    
    stopwords = [" and", " of", " or", " so"] + punctuations
    stopwords_ids = [tokenizer.encode(word)[0] for word in stopwords]

    locate_ixes=[]
    for i, attn in enumerate(cls_attns):
        
        current_sent = batch["input_ids"][i][: lengths[i]]
        # print("current_sent", current_sent)
        no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids).to(torch.device('cuda'))))[0]
        # print("no_punc_indices", no_punc_indices)
        
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
        # print("top_masks", top_masks)
        
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
                if (tokenizer.decode(current_sent[j]).strip() not in punctuations) and (tokenizer.decode(current_sent[j]) not in ['\n', ' ']):
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

# # model 의 forward 함수에서 정의를 output_attentions=True를 넘길 수 있게 되어 있다.
# def locate_embed(model, tokenizer, batch, max_num_tokens = 6, num_layer=10):
#     # torch.cuda.empty_cache()
#     # forward
#     model.eval()
#     with torch.no_grad():
#         classifier_output = model.forward(**batch, output_attentions=True)
#         torch.cuda.empty_cache()
        
#     # get attentions
#     attentions = classifier_output["attentions"]
#     # attention_mask에서 1의 개수를 셈
#     # lengths = [i.tolist().count(1) for i in batch["attention_mask"]]
#     lengths = [len(x) for x in batch["input_ids"]]
#     # 보고자 하는 attention layer 만 가져옴
#     attentions = attentions[
#         num_layer # originally 10
#     ]
#     cls_attns = attentions.max(1)[0][:, 0]
    
#     stopwords = [" and", " of", " or", " so"] + punctuations
#     stopwords_ids = [tokenizer.encode(word)[0] for word in stopwords]

#     locate_ixes=[]
#     for i, attn in enumerate(cls_attns):
        
#         current_sent = batch["input_ids"][i][: lengths[i]]
#         # print("current_sent", current_sent)
#         no_punc_indices = torch.where(~torch.isin(current_sent, torch.tensor(stopwords_ids).to(torch.device('cuda'))))[0]
#         # print("no_punc_indices", no_punc_indices)
        
#         # current tokenizer does not add <s> and </s> to the sentence.
#         current_attn = attn[: lengths[i]].softmax(-1) 
#         # print("current_attn", current_attn)
#         current_attn = current_attn[no_punc_indices]
#         # print("current_attn", current_attn)
        
#         # 이 값의 평균을 구한다.
#         avg_value = current_attn.view(-1).mean().item()
#         # print("avg_value", avg_value)
#         # 이 값 중에 평균보다 큰 값을 지니는 위치를 찾는다.
#         # fixed to reflect that sometimes the sequence length is 1.
#         top_masks = ((current_attn >= avg_value).nonzero().view(-1)) 
#         torch.cuda.empty_cache()
#         top_masks = top_masks.cpu().tolist()
#         # print("top_masks", top_masks)
        
#         # attention 값이 평균보다 큰 토큰의 수가 k개 또는 문장 전체 토큰 수의 1/3 보다 크면  
#         if len(top_masks) > min((lengths[i]) // 3, max_num_tokens):
#             # 그냥 attention 값 기준 k 개 또는 토큰 수/3 중 작은 수를 뽑는다.
#             top_masks = (
#                 current_attn.topk(max(min((lengths[i]) // 3, max_num_tokens), 1))[1]
#             )
#             top_masks = top_masks.cpu().tolist()
#             # print("top k top_masks", top_masks)
#         top_masks_final = no_punc_indices[top_masks]
#         # print("top_masks_final", top_masks_final)
        
#         # word의 일부만 locate 한 경우, word 전체를 locate 한다.
#         # 같은 word 안에 있는 token 끼리 묶음.
#         j, k = 0, 0
#         grouped_tokens = []
#         grouped_tokens_for_word = []
#         words = tokenizer.decode(current_sent).strip().split()
#         # print("words", words)
#         while j < len(current_sent):
#             if (tokenizer.decode(current_sent[j]).strip() not in punctuations) and (tokenizer.decode(current_sent[j]) not in ['\n', ' ']):
#                 # print("tokenizer.decode(current_sent[j])", tokenizer.decode(current_sent[j]))
#                 while k < len(words):
#                     if tokenizer.decode(current_sent[j]).strip() in words[k]:
#                         grouped_tokens_for_word.append(j)
#                         break
#                     else:
#                         grouped_tokens.append(grouped_tokens_for_word)
#                         grouped_tokens_for_word = []
#                         k += 1
#             j += 1
#         grouped_tokens.append(grouped_tokens_for_word)
#         # print("grouped_tokens", grouped_tokens)
        
#         top_masks_final.sort()
#         top_masks_final_final = []
#         for index in top_masks_final:
#             # print("index", index)
#             if index not in top_masks_final_final:
#                 word = [grouped_ixes for grouped_ixes in grouped_tokens if index in grouped_ixes]
#                 # print("word", word)
#                 if len(word) > 0:
#                     word = word[0]
#                 else:
#                     print(f"!!! {index} not in the grouped_ixes {grouped_tokens}")
#                     print(f"!!! tokenizer.decode(index): {tokenizer.decode(index)}")
#                 top_masks_final_final.extend(word)
#         locate_ixes.append(list(set(top_masks_final_final)))

            
#     return locate_ixes