import torch
import string

punctuations = string.punctuation + '\n '
punctuations = list(punctuations)
punctuations.remove('-')


# model 의 forward 함수에서 정의를 output_attentions=True를 넘길 수 있게 되어 있다.
def locate(model, tokenizer, batch, max_num_tokens = 6, num_layer=10):
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
    
    locate_ixes=[]
    for i, attn in enumerate(cls_attns):
        # attention_mask가 1인 곳 까지의 attention을 보고, start of sentence와 end of sentence에 해당하는 token을 제거하고, softmax를 취한다.
        # current_attn = attn[: lengths[i]][1:-1].softmax(-1)
        current_attn = attn[: lengths[i]].softmax(-1) # <- current tokenizer does not add <s> and </s> to the sentence.
        # 이 값의 평균을 구한다.
        avg_value = current_attn.view(-1).mean().item()
        # 이 값 중에 평균보다 큰 값을 지니는 위치를 찾는다. (+1 because we skipped the first token)
        # top_masks = ((current_attn >= avg_value).nonzero().view(-1)) + 1
        top_masks = ((current_attn >= avg_value).nonzero().view(-1)) ## fixed to reflect that sometimes the sequence length is 1.
        torch.cuda.empty_cache()
        top_masks = top_masks.cpu().tolist()
        if len(current_attn)==1:
            print(current_attn, top_masks, lengths[i])
        
        # ToDo: stopwords 제거를 먼저하고, 그 다음에 topk를 뽑는다.
        
        # attention 값이 평균보다 큰 토큰의 수가 6 또는 문장 전체 토큰 수의 1/3 보다 크면  
    #     if len(top_masks) > min((lengths[i] - 2) // 3, 6):
        if len(top_masks) > min((lengths[i]) // 3, max_num_tokens):
            # 그냥 attention 값 기준 top k 개 (k = 6 또는 토큰 수/3)를 뽑는다.
            top_masks = (
                # current_attn.topk(min((lengths[i] - 2) // 3, 6))[1] + 1
    #             current_attn.topk(min((lengths[i] - 2) // 3, 6))[1]
                current_attn.topk(max(min((lengths[i]) // 3, max_num_tokens), 1))[1]
            )
            top_masks = top_masks.cpu().tolist()
            if len(current_attn) == 1:
                print("top_masks after selecting top k", top_masks)
        # 현재 문장의 input id를 가져온다.
        current_sent = batch["input_ids"][i][: lengths[i]]
        count = 0
        top_masks_final = []
        # top_masks에 뽑힌 index를 돌면서
        for index in top_masks:
            # mask해야 할 토큰이 and, of, or, so 에 해당하지 않으면
            if tokenizer.decode(current_sent[index]) not in [ ## maybe add more!
                " and",
                " of",
                " or",
                " so",
            ] + punctuations : # added punctuations
                # token을 mask 한다.
                # current_sent[index] = mask_token
                top_masks_final.append(index)
                # count 수를 늘린다.
                count += 1
            else:
                # 만약에 and, of, or, so 에 해당하면 아무것도 하지 않는다.
                pass
        locate_ixes.append(top_masks_final)
    return locate_ixes