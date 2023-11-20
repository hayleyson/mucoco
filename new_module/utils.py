import os
import logging
from typing import List

import torch
import torch.nn.functional as F
import transformers

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'DEBUG').upper(), 
                    format='%(message)s')
logger = logging.getLogger(__name__)

def lm_nll_loss(prefix_ids: torch.Tensor, target_ids: torch.Tensor, lm: transformers.AutoModel, length_normalize_yn: bool = False) -> float:
    """ Function to calculate nll loss for next token prediction task
    @param prefix_ids (Tensor): token ids of prompt. tensor of shape (batch_size, sequence_length)
    @param target_ids (Tensor): token ids of target continuation. tensor of shape (batch_size, sequence_length)
    @param lm (AutoModel): transformers AutoModel object to use for calculating nll
    @param length_normalize_yn (bool): whether to normalize loss by sequence length

    @returns loss (float): length-normalized nll loss
    """

    prefix_ids = prefix_ids.squeeze(0)
    target_ids = target_ids.squeeze(0)
    
    with torch.no_grad():
        lm_output = lm(input_ids = torch.cat([prefix_ids, target_ids], dim=-1))

    labels = target_ids
    logp = F.log_softmax(lm_output.logits[prefix_ids.shape[-1]-1:, :], dim=-1)

    # required arguments to F.nll_loss : input, target
    # input's shape: (sequence_length, num_classes)
    # target's shape: (num_classes) 
    loss = F.nll_loss(logp[:-1], labels, reduction="none").sum(dim=-1).item()

    if length_normalize_yn == True:
        loss /= labels.size(-1)

    torch.cuda.empty_cache()

    return loss


def score_hypotheses_lm(my_prefix: torch.Tensor, 
                        my_hypotheses: List[torch.Tensor], 
                        primary_model: transformers.AutoModel, 
                        length_normalize_yn: bool = False):
    total_loss_list = []
    for hyp in my_hypotheses:
        total_loss_list.append(lm_nll_loss(my_prefix, hyp.unsqueeze(0), primary_model, length_normalize_yn))
    return total_loss_list


def score_hypotheses(my_prefix: torch.Tensor, my_hypotheses, my_config, mylossfns, **kwargs):

    total_loss_list = []
    primr_loss_list = []
    logging_loss_list = []
    for hyp in my_hypotheses:
        curr_loss = 0
        logging_loss = []
        for _lossid, _lossname in enumerate(my_config['losses']):
            with torch.no_grad():
                lossvalue, logging_output =\
                    mylossfns[_lossid].compute_gold_loss(
                        (my_prefix, hyp.unsqueeze(0)), 
                        additional_batch=kwargs['additional_batch'], 
                        context_batch=kwargs['context_batch'],
                        use_context=kwargs['use_context'],
                        label_id=kwargs['label_ids'][_lossid],
                        keyword=kwargs['keywords'][_lossid],
                        kweight=kwargs['kweight']
                    )
            if _lossid == 0:
                primr_loss_list.append(lossvalue.item())
            logging_loss.append(lossvalue.item())
            curr_loss += my_config['loss_weights'][_lossid] * lossvalue.item()
        total_loss_list.append(curr_loss)
        logging_loss_list.append(logging_loss)
    return total_loss_list, primr_loss_list, logging_loss_list


def constrained_beam_search(source_batch,
                            masked_sequence, 
                            mask_token_index_primary, 
                            primary_mask_token_id, 
                            predicted_token_ids, 
                            primary_tokenizer, 
                            mlm_tokenizer, 
                            primary_model, 
                            config, 
                            beam_size = 5):
    
    hypotheses = [torch.LongTensor([]).to(config['device'])]
    
    L = masked_sequence.size(-1)
    # logger.debug(mask_token_index_primary)
    # mask_token_index_primary = mask_token_index - 1 # to account for <sos> token attached at the beginning by mlm tokenizer
    
    masked_count = 0
    for i in range(L):
        # logger.debug(f"index {i}")
        if masked_sequence[0, i] != primary_mask_token_id:
            for j in range(len(hypotheses)):
                hypotheses[j] = torch.cat([hypotheses[j], masked_sequence[0, i].unsqueeze(0)], dim = -1)
            # logger.debug(f"hypotheses at {i}: {hypotheses}")
        else:
            hypotheses_exp = []
            for hyp in hypotheses:
                # logger.debug(f"hyp: {hyp}")
                for j in range(config['k_per_location']):
                    candidate = predicted_token_ids.indices[torch.where(mask_token_index_primary == i)[0], j]
                    # logger.debug(f"candidate: {candidate}")
                    candidate = primary_tokenizer.encode(mlm_tokenizer.decode(candidate), return_tensors="pt").to(config['device'])
                    # logger.debug(f"candidate: {candidate}")
                    # logger.debug(f"candidate.squeeze(): {candidate.squeeze()}")
                    hypotheses_exp.append(torch.cat([hyp, candidate.squeeze(0)], dim=-1))
    
            # logger.debug(f"hypotheses_exp at {i}: {hypotheses_exp}")

            # logger.debug(f"source_batch: {source_batch}")
            losses =  score_hypotheses_lm(
                source_batch, 
                hypotheses_exp, 
                primary_model,
                config['build_loss_dict']['length_normalize']
            )
            # logger.debug(f"losses: {losses}")
    
            hypotheses = sorted(zip(hypotheses_exp, losses), key=lambda x: x[1])[:beam_size]
            hypotheses = [x[0] for x in hypotheses]
            
    return hypotheses



def constrained_beam_search_v0(source_batch,
                               masked_sequence, 
                               mask_token_index_primary, 
                               primary_mask_token_id, 
                               predicted_token_ids, 
                               primary_tokenizer, 
                               mlm_tokenizer, 
                               lossfns,
                               config, 
                               beam_size = 5,
                              **kwargs):
    hypotheses = [torch.LongTensor([]).to(config['device'])]
    
    L = masked_sequence.size(-1)
    # logger.debug(mask_token_index_primary)
    # mask_token_index_primary = mask_token_index - 1 # to account for <sos> token attached at the beginning by mlm tokenizer
    
    masked_count = 0
    for i in range(L):
        # logger.debug(f"index {i}")
        if masked_sequence[0, i] != primary_mask_token_id:
            for j in range(len(hypotheses)):
                hypotheses[j] = torch.cat([hypotheses[j], masked_sequence[0, i].unsqueeze(0)], dim = -1)
            # logger.debug(f"hypotheses at {i}: {hypotheses}")
        else:
            hypotheses_exp = []
            for hyp in hypotheses:
                # logger.debug(f"hyp: {hyp}")
                for j in range(config['k_per_location']):
                    candidate = predicted_token_ids.indices[torch.where(mask_token_index_primary == i)[0], j]
                    # logger.debug(f"candidate: {candidate}")
                    candidate = primary_tokenizer.encode(mlm_tokenizer.decode(candidate), return_tensors="pt").to(config['device'])
                    # logger.debug(f"candidate: {candidate}")
                    # logger.debug(f"candidate.squeeze(): {candidate.squeeze()}")
                    hypotheses_exp.append(torch.cat([hyp, candidate.squeeze(0)], dim=-1))
    
            # logger.debug(f"hypotheses_exp at {i}: {hypotheses_exp}")
    
            losses, primary_losses, constraint_losses = score_hypotheses(source_batch,
                                                                         hypotheses_exp, 
                                                                         config, 
                                                                         lossfns,
                                                                         additional_batch=kwargs['additional_batch'], 
                                                                        context_batch=kwargs['context_batch'],
                                                                        use_context=kwargs['use_context'],
                                                                        label_ids=kwargs['label_ids'],
                                                                        keywords=kwargs['keywords'],
                                                                        kweight=kwargs['kweight'])
    
            hypotheses = sorted(zip(hypotheses_exp, losses), key=lambda x: x[1])[:beam_size]
            hypotheses = [x[0] for x in hypotheses]
            
    return hypotheses


def editing_beam_search(
                    source_batch: torch.Tensor,
                    predicted_batch: torch.Tensor, 
                    edit_token_index_primary, 
                    primary_model: transformers.AutoModel, 
                    primary_tokenizer: transformers.AutoTokenizer,
                    config: dict, 
                    beam_size: int
                ) -> torch.Tensor:
    """ Function that autoregressively edits a sequence(predicted_batch) by updating tokens at edit_token_index_primary indices and keeping the other tokens as were.
    @param source_batch (Tensor): token ids of the prefix
    @param predicted_batch (Tensor): token ids of the original continuation
    @param edit_token_index_primary (Tensor): indices that indicate locations in the original continuation to edit
    @param primary_model (AutoModel): model to calculate likelihood of candidate sequences
    @param primary_tokenizer (AutoTokenizer): tokenizer for the primary_model
    @param config (dict)
    @param beam_size (int)

    @returns hypotheses (Tensor): beam_size number of hypotheses to edit the original continuation. Tensor of shape (beam_size, sequence length).
    """

    hypotheses = torch.LongTensor([[]]).to(config['device'])
    hyp_scores = torch.zeros(len(hypotheses), dtype = torch.float, device = config['device'])
    seq_len = predicted_batch.size(-1)
    
    for t in range(seq_len):
        
        prefix_added_hypotheses = torch.cat([source_batch.expand(hypotheses.size(0), -1), hypotheses], dim=-1)
        with torch.no_grad():
            model_output = primary_model(input_ids = prefix_added_hypotheses)

        logits_t = model_output.logits[:, -1, :] # get logits for the last timestep
        logp_t = F.log_softmax(logits_t, dim=-1) # (num_hypotheses, |V|)
        
        if t not in edit_token_index_primary:
            
            curr_nll = F.nll_loss(logp_t, predicted_batch[:, t].expand(logp_t.size(0)), reduction="none") # returns (num_hypotheses)
            hyp_scores = hyp_scores.expand_as(curr_nll) + curr_nll # (num_hypotheses)
            hypotheses = torch.cat([hypotheses, predicted_batch[:, t].expand(hypotheses.size(0), -1)], dim=-1)
            
        else:
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(logp_t) + (-logp_t)).view(-1) # (num_hypotheses x |V|)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=beam_size, largest=True)
            
            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(primary_tokenizer), rounding_mode='floor') # prev_hyp_id for each of top_cand_hyp. (beam_size)
            hyp_word_ids = top_cand_hyp_pos % len(primary_tokenizer) # hyp_word_id for each of top_cand_hyp. (beam_size)
            
            hypotheses = torch.cat([hypotheses[prev_hyp_ids], hyp_word_ids.unsqueeze(1)], dim=-1)
            hyp_scores = top_cand_hyp_scores

        torch.cuda.empty_cache()
    return hypotheses