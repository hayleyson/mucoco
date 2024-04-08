import logging
import os
from typing import List

import torch
import torch.nn.functional as F
import transformers

import wandb

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'DEBUG').upper(), 
                    format='%(message)s')
logger = logging.getLogger(__name__)

def beam_rerank_v0(source_text, ## text (too arbitrary?)
                    masked_sequence, ## in mlm tokenizer's tokens
                    indices_in_mlm_tokens,
                    predicted_token_ids,
                    mlm_tokenizer, 
                    lossfns,
                    config, 
                    beam_size = 5):
    
    hypotheses = [torch.LongTensor([]).to(config['device'])]
    L = masked_sequence.size(-1)

    for i in range(L):
        if masked_sequence[0, i] != mlm_tokenizer.mask_token_id:
            hypotheses = list(torch.cat([torch.stack(hypotheses,dim=0), 
                                        masked_sequence[:, i].unsqueeze(0).repeat((len(hypotheses),1)).to(config['device'])],dim=-1))
        else:
            num_hypotheses = len(hypotheses)
            hypotheses = torch.stack(hypotheses,dim=0).unsqueeze(0)
            hypotheses = hypotheses.repeat(config['k_per_location'], 1, 1)
            candidates = predicted_token_ids.indices[torch.where(indices_in_mlm_tokens == i)[0], :].to(config['device']).T.unsqueeze(1)
            candidates = candidates.repeat(1, num_hypotheses, 1)
            hypotheses_exp = torch.cat([hypotheses, candidates], dim=-1)
            hypotheses_exp = hypotheses_exp.view(-1, hypotheses_exp.shape[-1])
            hypotheses_exp = list(hypotheses_exp)

            losses = []
            # loss_weights = [1 - wandb.config.closs_weight, wandb.config.closs_weight]
            loss_weights = config['loss_weights']
            for hyp in hypotheses_exp:
                curr_loss = 0.0
                for lossid, lossname in enumerate(config["losses"]):
                    with torch.no_grad():
                        lossvalue = lossfns[lossid].compute_gold_loss(
                            source_text, mlm_tokenizer.decode(hyp, skip_special_tokens=True),
                            label_id=config['target_label_ids'][lossid],
                        )
                    curr_loss += loss_weights[lossid] * lossvalue.item()
                losses.append(curr_loss)

            hypotheses = sorted(zip(hypotheses_exp, losses), key=lambda x: x[1])[:beam_size]
            hypotheses = [x[0] for x in hypotheses]
            
    return [mlm_tokenizer.decode(x, skip_special_tokens=True) for x in hypotheses]


def beam_rerank_v1(source_text, ## text (too arbitrary?)
                    masked_sequence, ## in mlm tokenizer's tokens
                    indices_in_mlm_tokens,
                    predicted_token_ids,
                    mlm_tokenizer, 
                    lossfns,
                    config, 
                    beam_size = 5):
    
    hypotheses = [torch.LongTensor([]).to(config['device'])]
    L = masked_sequence.size(-1)

    for i in range(L):
        if masked_sequence[0, i] != mlm_tokenizer.mask_token_id:
            hypotheses = list(torch.cat([torch.stack(hypotheses,dim=0), 
                                        masked_sequence[:, i].unsqueeze(0).repeat((len(hypotheses),1)).to(config['device'])],dim=-1))
        else:
            num_hypotheses = len(hypotheses)
            hypotheses = torch.stack(hypotheses,dim=0).unsqueeze(0)
            hypotheses = hypotheses.repeat(config['k_per_location'], 1, 1)
            candidates = predicted_token_ids.indices[torch.where(indices_in_mlm_tokens == i)[0], :].to(config['device']).T.unsqueeze(1)
            candidates = candidates.repeat(1, num_hypotheses, 1)
            hypotheses_exp = torch.cat([hypotheses, candidates], dim=-1)
            hypotheses_exp = hypotheses_exp.view(-1, hypotheses_exp.shape[-1])
            hypotheses_exp = list(hypotheses_exp)

            losses = []
            for hyp in hypotheses_exp:
                with torch.no_grad():
                    lossvalue = lossfns[0].compute_gold_loss(
                        source_text, mlm_tokenizer.decode(hyp, skip_special_tokens=True)
                    )
                losses.append(lossvalue.item())

            hypotheses = sorted(zip(hypotheses_exp, losses), key=lambda x: x[1])[:beam_size]
            hypotheses = [x[0] for x in hypotheses]
            
    return [mlm_tokenizer.decode(x, skip_special_tokens=True) for x in hypotheses]


def beam_rerank_v2(
    source_batch: torch.Tensor, ## prompt in primary tokenizer's tokens
    masked_sequence: torch.Tensor, ## masked sequence in primary tokenizer's tokens
    primary_model: transformers.AutoModel, 
    primary_tokenizer: transformers.AutoTokenizer,
    config: dict, 
    beam_size: int
):
    hypotheses = torch.LongTensor([[]]).to(config['device'])
    hyp_scores = torch.zeros(len(hypotheses), dtype = torch.float, device = config['device'])
    L = masked_sequence.size(-1)

    for t in range(L):
        prefix_added_hypotheses = torch.cat([source_batch.expand(hypotheses.size(0), -1), hypotheses], dim=-1)
        # print(prefix_added_hypotheses)
        with torch.no_grad():
            model_output = primary_model(input_ids = prefix_added_hypotheses)

        logits_t = model_output.logits[:, -1, :] # get logits for the last timestep
        logp_t = F.log_softmax(logits_t, dim=-1) # (num_hypotheses, |V|)
        
        if masked_sequence[:,t] != primary_tokenizer.mask_token_id:
            
            curr_nll = F.nll_loss(logp_t, masked_sequence[:, t].expand(logp_t.size(0)), reduction="none") # returns (num_hypotheses)
            hyp_scores = hyp_scores.expand_as(curr_nll) + curr_nll # (num_hypotheses)
            hypotheses = torch.cat([hypotheses, masked_sequence[:, t].expand(hypotheses.size(0), -1)], dim=-1)
            
        else:
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(logp_t) + (-logp_t)).view(-1) # (num_hypotheses x |V|)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=beam_size, largest=True)
            
            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(primary_tokenizer), rounding_mode='floor') # prev_hyp_id for each of top_cand_hyp. (beam_size)
            hyp_word_ids = top_cand_hyp_pos % len(primary_tokenizer) # hyp_word_id for each of top_cand_hyp. (beam_size)
            
            hypotheses = torch.cat([hypotheses[prev_hyp_ids], hyp_word_ids.unsqueeze(1)], dim=-1)
            hyp_scores = top_cand_hyp_scores

        torch.cuda.empty_cache()
    return [primary_tokenizer.decode(x, skip_special_tokens=True) for x in list(hypotheses)]

def combi_rerank(masked_sequence, ## in mlm tokenizer's tokens
                 indices_in_mlm_tokens,
                 predicted_token_ids,
                 mlm_tokenizer,
                 config):
    ## get k ** num_located_indices sequences with different combinations of the top k tokens for located locations
    ## hypotheses will hold a list of texts
    hypotheses = []
    num_located_tokens = len(indices_in_mlm_tokens)
    num_all_cases = config["k_per_location"] ** num_located_tokens
    tok_cand_combo = [0 for i in range(num_located_tokens)]

    for case_id in range(num_all_cases):
        for i in range(num_located_tokens):
            tok_cand_combo[i] = (
                case_id // (config["k_per_location"] ** i)
            ) % config["k_per_location"]

        tmp_seq = masked_sequence.clone()
        for pos_id, tok_cand_id in enumerate(tok_cand_combo):
            tmp_seq[
                0, indices_in_mlm_tokens[pos_id]
            ] = predicted_token_ids.indices[pos_id, tok_cand_id]

        tmp_dec_seq = mlm_tokenizer.batch_decode(
                tmp_seq, skip_special_tokens=True
        )
        hypotheses.append(tmp_dec_seq[0])
    return hypotheses