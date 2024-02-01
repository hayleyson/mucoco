#!/usr/bin/env python
# coding: utf-8


# standard libraries
import os
import sys
import json
import logging
import argparse
import time 
from collections import namedtuple
from typing import List

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../" )
sys.path.append(project_dir)
os.chdir(project_dir)
print("project_dir: ", project_dir)
print("current_dir: ", os.getcwd())
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

# installed packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
import wandb
import random

# custom libraries
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, OptimizerLE, get_epsilon, locate
from new_module.evaluate_wandb import evaluate
from new_module.decode_utils import score_hypotheses, constrained_beam_search_v0, constrained_beam_search, editing_beam_search


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGGING_LEVEL', logging.DEBUG))
# logger.setLevel(logging.INFO)

def main(config):
    
    main_start_time = time.time()
    
    if not config.get('model_tag', None):
        if 'energy-training' in config['model_paths'][1]:
            config['model_tag'] = "em"
        else:
            config['model_tag'] = "clsf"
            
        if (config['task'] == 'formality') and ('gyafc' in config['model_paths'][1]):
            config['model_tag']+="-gyafc"
    
    if config['resume']:
        logger.info("resuming from a previous run")
        run = wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], id=config['wandb_run_id'], resume='must')
    else:
        run = wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)
     
    run_id=run.path.split('/')[-1]
    display_name = f"{config['method']}-{config['locate_unit']}-nps{wandb.config.num_edit_token_per_step}-k{wandb.config.k_per_location}-beam{wandb.config.beam_size}-{wandb.config.selection_criteria}-{run_id}"
    display_name += f"-{config['source_style']}-to-{config['target_style']}"
    # if config['task'] == 'formality':
    #     display_name += f"-{config['source_style']}-to-{config['target_style']}"
    # elif config['task'] == 'sentiment':
    #     display_name += f"-{config['target_style']}"
    
    outdir = os.path.join(config['output_dir_prefix'], display_name)
    os.makedirs(outdir, exist_ok=True)
    outfile = f"{outdir}/outputs_epsilon{config['min_epsilons'][0]}.txt"
    run.summary['outfile_path'] = outfile
   
    class dummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    build_loss_args=dummyArgs(**config['build_loss_dict'])
    
    ## load data
    if (config['task'] == 'toxicity') or (config['task'] == 'sentiment'):
        source_dataset = [json.loads(l)[config['jsonl_primary_key']][config['jsonl_secondary_key']] for l in open(config['source_data'])]
        generation_dataset = [json.loads(l)["generations"] for l in open(config['source_data'])]
    elif (config['task'] == 'formality') or (config['task']=='sentiment-lewis-compr'):
        with open(config['source_data'],'r') as f:
            generation_dataset = f.readlines()
        source_dataset = ["" for l in generation_dataset] 
    
    
    #check if outfile exists
    if (config['resume']) and (os.path.exists(outfile)):
    # if os.path.exists(outfile):
        
        with open(outfile, "r") as f:
            existing_gens = [x.rstrip('\n') for x in f.readlines()]
        resume_idx = len(existing_gens)
        if resume_idx == len(source_dataset):
            logger.debug(f"output file is already complete. skipping this run.")
            return
        elif resume_idx < len(source_dataset):
            logger.info(f"output file already exists but is incomplete. resuming from index: {resume_idx}")
            outf = open(outfile, "a")
        else:
            logger.critical(f"output file seems to be corrupted. The file length is {resume_idx}, where the size of source_dataset is {len(source_dataset)}")
            return
    else:
        resume_idx = 0
        outf = open(outfile, "w")
    
    ## load tokenizer, models, define losses
    name2tokenizer = {}
    name2model = {}
    name2config = {}
    loss2modelname = {}
    loss2tokenizer = {}
    embed_luts = []
    embed_scales = []
    prev_vocab_size = None
    vocab_size = None
    primary_vocab_size = None
    primary_model = None

    for i, model_path in enumerate(config['model_paths']):
        if model_path not in name2model: #making sure we are not loading the model twice in case some constraints use the same model. 
            
            try:
                name2tokenizer[model_path] = AutoTokenizer.from_pretrained(config['tokenizer_paths'][i], cache_dir=config['cache_dir'],  use_fast=True)
            except:
                name2tokenizer[model_path] = AutoTokenizer.from_pretrained(config['tokenizer_paths'][i], cache_dir=config['cache_dir'],  use_fast=False)
                
            name2config[model_path] = AutoConfig.from_pretrained(model_path, cache_dir=config['cache_dir'])

            if config['model_types'][i] == "sentence-transformer":
                name2model[model_path] = lossbuilder.ModelWrapper(SentenceTransformer(model_path))
            elif "Custom" in config['model_types'][i]:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(utils, config['model_types'][i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=config['cache_dir']))
            else:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(transformers, config['model_types'][i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=config['cache_dir']))
            name2model[model_path].eval()
            name2model[model_path].cuda()
            embed_lut_ = name2model[model_path].get_input_embeddings()
            if isinstance(embed_lut_, torch.nn.Sequential):
                new_vocab_size = embed_lut_[0].num_embeddings
            else:
                new_vocab_size = embed_lut_.num_embeddings
            if prev_vocab_size is None:
                vocab_size=new_vocab_size
            prev_vocab_size = vocab_size
        
        input_embeds = name2model[model_path].get_input_embeddings()
        if isinstance(input_embeds, torch.nn.Sequential):
            input_embeds = input_embeds[0]
        embed_luts.append(input_embeds)
        
        if config['target_type'] == "embeds":
            embed_luts[-1].requires_grad=False
        
        if i == 0:
            primary_vocab_size = vocab_size
            primary_embed_dim = embed_luts[-1].embedding_dim
            primary_model = name2model[model_path]
        
        if getattr(name2model[model_path], "get_decoder", None) is None: #this is for MarianMT models which have a weird embedding_scale parameter
            embed_scales.append(1.0)
        else:
            embed_scales.append(getattr(name2model[model_path].get_decoder(), "embed_scale", 1.0))

    lossfns = []
    for i, loss in enumerate(config['losses']):
        lossfns.append(lossbuilder.build_loss(loss, name2model[config['model_paths'][i]], name2tokenizer[config['model_paths'][i]], build_loss_args))
        loss2modelname[loss] = config['model_paths'][i]
        loss2tokenizer[loss] = name2tokenizer[config['model_paths'][i]]
    primary_tokenizer = loss2tokenizer[config['losses'][0]]
    
    ## load model to generate candidates for editing
    if config['method'] == "mlm-beamsearch-v2":
        mlm_tokenizer = None
        mlm = None
        
    else:
        model_checkpoint = "roberta-base"
        mlm_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        mlm = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

        primary_tokenizer.add_special_tokens({'mask_token':mlm_tokenizer.mask_token})
        primary_mask_token_id = primary_tokenizer.mask_token_id
        
    run.summary['prep_time'] = time.time() - main_start_time
    ## beginning of main logic
    decode_start_time = time.time()
    # text_id = 0
    if config['resume']:
        num_skipped = run.summary.get('num_skipped', 0)
        num_edited = run.summary.get('num_edited',0)
    else:
        num_skipped = 0
        num_edited = 0
        
    for text_id in range(len(source_dataset))[resume_idx:]:
        source_text = source_dataset[text_id]
        if source_text == "":
            source_text = primary_tokenizer.bos_token
        source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(config['device']).long()
        source_batch = torch.cat([source_indices], dim=0).to(config['device'])
        
        if (config['task'] == 'toxicity') or (config['task'] == 'sentiment'):
            predicted_batches = [x["tokens"] for x in generation_dataset[text_id]]
            predicted_batches = [torch.tensor([x], dtype=torch.long, device=config['device']) for x in predicted_batches]
            AR_prediction_all = [x["text"] for x in generation_dataset[text_id]]
        elif (config['task'] == 'formality') or (config['task']=='sentiment-lewis-compr'):
            predicted_batches = primary_tokenizer.encode(generation_dataset[text_id], return_tensors="pt", add_special_tokens=False).to(config['device']).unsqueeze(0)
            AR_prediction_all = [generation_dataset[text_id]]
        
        sample_idx = 0
        for sample_idx in range(config['num_samples'])[:]:
            predicted_batch = predicted_batches[sample_idx].cuda()
            AR_prediction = AR_prediction_all[sample_idx]
            
            logger.debug(f"text_id {text_id} sample_id {sample_idx} \n[prompt] {source_text} [text] {AR_prediction}")# \n[input_ids] {predicted_batch}")
            # logger.critical(f"text_id {text_id} sample_id {sample_idx} \n[text] {AR_prediction} \n[input_ids] {predicted_batch}")
            # logger.critical(predicted_batch.shape)
            # logger.critical(source_batch.shape)
            
            ## check if toxicity less than threshold
            gold_losses = []
            label_ids = config['target_label_ids'] # target label's ids for each loss
            keywords = ["the" for _ in config['losses']]
            new_kweight = 5.0
            use_context = 'false'
            allsat = True
            additional_batch = source_batch
            context_batch = [None]
            gold_losses = []
            for lossid, lossname in enumerate(config['losses']):
                with torch.no_grad():
                    lossvalue, logging_output =\
                        lossfns[lossid].compute_gold_loss(
                            # (source_batch, target_batch), # bug: if it's target_batch, we're inputting 2 copies of source_batch
                            (source_batch, predicted_batch), 
                            additional_batch=additional_batch, 
                            context_batch=context_batch,
                            use_context=use_context,
                            label_id=label_ids[lossid],
                            keyword=keywords[lossid],
                            kweight=new_kweight,
                            # primary_tokenizer=primary_tokenizer,
                            # device=config['device']
                        )
                gold_losses.append(lossvalue.squeeze().item())
                
                if (lossid >= 1):
                    if gold_losses[lossid] > -np.log(config['min_epsilons'][lossid - 1]):
                        allsat = False
                    # # if (label_ids[lossid] == 0) and (gold_losses[lossid] > config['min_epsilons'][lossid - 1]): ## loss must be less than epsilon
                    # if (label_ids[lossid] == 0) and (gold_losses[lossid] < -np.log(config['min_epsilons'][lossid - 1])): ## loss must be less than epsilon
                    #     allsat = False
                    # # elif (label_ids[lossid] == 1) and (gold_losses[lossid] < config['min_epsilons'][lossid - 1]): ## loss must be greater than epsilon
                    # elif (label_ids[lossid] == 1) and (gold_losses[lossid] > -np.log(config['min_epsilons'][lossid - 1])): ## loss must be greater than epsilon
                    #     allsat = False
            if allsat:
                logger.info(f"skipping this sample since it already satisfies constraint. {gold_losses}")
                num_skipped += 1
                if sample_idx == 0:
                    output = {
                        "prompt":{
                            "text":source_text,
                            "tokens":source_indices.tolist()
                            }, 
                        "generations":[{
                            "text": "",
                            "tokens": [],
                            "indices": [[]], 
                            "allsat": -1,
                            "losses": gold_losses,
                            "weighted_loss": -1
                            }]
                    }
                else:
                    output['generations'].append(
                        {
                            "text": "",
                            "tokens": [],
                            "indices": [[]], 
                            "allsat": -1,
                            "losses": gold_losses,
                            "weighted_loss": -1
                        }
                    )
            
                if sample_idx + 1 == config['num_samples']:
                    json.dump(output, outf)
                    outf.write("\n")
                    outf.flush()
                
            else:
                
                num_edited += 1
                es_patience_count = 0
                original_sequence = None
                best_ix, best_prediction, best_text, best_allsat, best_losses, best_weighted_loss = None, None, None, None, None, None
                
                _iter = 0
                for _iter in range(wandb.config.n_iter):
                    
                    ## locate tokens to edit
                    batch = {"input_ids": predicted_batch}
                    original_sequence = predicted_batch
                    indices = locate(name2model[config['model_paths'][1]], 
                                    name2tokenizer[config['model_paths'][1]], 
                                    batch, 
                                    max_num_tokens=wandb.config.num_edit_token_per_step, 
                                    unit=config['locate_unit'], 
                                    use_cuda=True)
                    logger.debug(f"iter {_iter}, sample_idx: {sample_idx}")
                    logger.debug(f"located indices: {indices}")
                    # logger.debug(f"located indices: {name2tokenizer[config['model_paths'][1]].decode(predicted_batch[:, indices].squeeze())}")
                    logger.debug(f"located indices: {name2tokenizer[config['model_paths'][0]].decode(predicted_batch[:, indices].squeeze())}")
                    # logger.debug(f"located indices: {predicted_batch[:, indices]}")
                    
                    if config['method'] == "mlm-beamsearch-v2":
                        pass
                    else:
                        ## replace tokens at the indices with mask tokens
                        
                        masked_sequence = predicted_batch.clone().detach()
                        masked_sequence[:, indices[0]] = primary_mask_token_id
                        masked_sequence_text = primary_tokenizer.batch_decode(masked_sequence.tolist())
                        # print("-"*50)
                        # print("original text")
                        # for i in range(len(predicted_batch[0])):
                        #     print(f"{i}: {predicted_batch[0][i]} | {primary_tokenizer.decode(predicted_batch[0][i])}", end=" ")
                        #     if i in indices[0]:
                        #         print(" -> located")
                        #     else:
                        #         print()
                        # print("-"*50) 
                        # print("masked_sequence_text: ", masked_sequence_text)
                        # print("-"*50)
                        
                        inputs = mlm_tokenizer(masked_sequence_text, return_tensors="pt")
                        
                        # ## c.f. check if spaces are preserved. -> preserved! checked.
                        # logger.debug(inputs['input_ids'])
                        # logger.debug(mlm_tokenizer.decode(inputs['input_ids'][0]))
                        
                        ## make predictions for the masked indices
                        with torch.no_grad():
                            logits = mlm(**inputs).logits
                        indices_in_mlm_tokens = (inputs.input_ids == mlm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                        
                        ## get top k tokens for each index 
                        predicted_token_ids = torch.topk(logits[0, indices_in_mlm_tokens], k=wandb.config.k_per_location, dim=-1)
                        # logger.debug(predicted_token_ids) # shape : (wandb.config.num_edit_token_per_step,  wandb.config.k_per_location)
                        
                        # logger.debug(predicted_token_ids)
                        # torch.return_types.topk(
                        # values=tensor([[16.153, 15.676, 15.537],
                        #         [14.131, 13.642, 13.477],
                        #         [12.802, 12.533, 12.361],
                        #         [19.653, 16.148, 15.160]]),
                        # indices=tensor([[32033,  1274,  5458],
                        #         [ 3993,   697,   342],
                        #         [   11,   106,    13],
                        #         [  106,    24,    15]]))

                        # logger.debug(f"masked_sequence: {masked_sequence} \n indices_in_mlm_tokens: {indices_in_mlm_tokens} \n \
                        #                primary_mask_token_id: {primary_mask_token_id} \n predicted_token_ids: {predicted_token_ids}")
                        # logger.debug(f"additional_batch={additional_batch}, source_batch = {source_batch}, context_batch={context_batch},\
                        # use_context={use_context},label_id={label_ids[lossid]},keyword={keywords[lossid]},kweight={new_kweight}")
                    
                    if config['method'] == "mlm-beamsearch-v0":
                        hypotheses = constrained_beam_search_v0(source_batch,
                                                                masked_sequence,
                                                                torch.LongTensor(sorted(indices[0])), 
                                                                primary_mask_token_id, 
                                                                predicted_token_ids, 
                                                                primary_tokenizer, 
                                                                mlm_tokenizer,
                                                                lossfns,
                                                                config, 
                                                                beam_size = wandb.config.beam_size,
                                                                additional_batch=additional_batch, 
                                                                context_batch=context_batch,
                                                                use_context=use_context,
                                                                label_ids=label_ids,
                                                                keywords=keywords,
                                                                kweight=new_kweight,
                                                                device=config['device']
                                                                )
                    elif config['method'] == "mlm-beamsearch-v1":    
                        hypotheses = constrained_beam_search(
                                        source_batch,
                                        masked_sequence, 
                                        torch.LongTensor(sorted(indices[0])), 
                                        primary_mask_token_id, 
                                        predicted_token_ids, 
                                        primary_tokenizer, 
                                        mlm_tokenizer, 
                                        primary_model, 
                                        config, 
                                        beam_size = wandb.config.beam_size
                                    )
                    elif config['method'] == "mlm-beamsearch-v2":
                        hypotheses = editing_beam_search(
                                            source_batch,
                                            predicted_batch, 
                                            torch.LongTensor(sorted(indices[0])), 
                                            primary_model, 
                                            primary_tokenizer,
                                            config, 
                                            beam_size = wandb.config.beam_size
                                        ) 
                    elif config['method'] == "mlm-reranking":
                        ## get k ** num_located_indices sequences with different combinations of the top k tokens for located locations
                        ## hypotheses will hold a list of input ids encoded with GPT2Tokenizer
                        hypotheses = []
                        num_located_tokens = len(indices[0])
                        num_all_cases = config['k_per_location'] ** num_located_tokens
                        tok_cand_combo = [0 for i in range(num_located_tokens)]
                        
                        for case_id in range(num_all_cases):
                            for i in range(num_located_tokens):
                                tok_cand_combo[i] = (case_id // (config['k_per_location']**i)) % config['k_per_location']
                            
                            tmp_seq = inputs['input_ids'].clone()
                            for pos_id, tok_cand_id in enumerate(tok_cand_combo):
                                tmp_seq[0, indices_in_mlm_tokens[pos_id]] = predicted_token_ids.indices[pos_id, tok_cand_id]
                            
                            # print("-"*50)
                            # print(f"Candidate {case_id} in RoBERTa tokens. Length: {len(tmp_seq[0])}")
                            # for i in range(len(tmp_seq[0])):
                            #     print(f"{i}: {tmp_seq[0][i]} | {mlm_tokenizer.decode(tmp_seq[0][i])}", end=" ")
                            #     if i in indices_in_mlm_tokens:
                            #         print(" -> located")
                            #     else:
                            #         print()
                            # print("-"*50) 
                            
                            # need to do decode with RobertaTokenizer and encode with GPT2Tokenizer
                            # logger.debug(mlm_tokenizer.batch_decode(tmp_seq[:, indices_in_mlm_tokens], skip_special_tokens=True))
                            tmp_dec_seq = primary_tokenizer(mlm_tokenizer.batch_decode(tmp_seq, skip_special_tokens=True), return_tensors="pt").input_ids.cuda()
                            hypotheses.append(tmp_dec_seq.squeeze(0))
                            
                            # print("-"*50)
                            # print(f"Candidate {case_id} in GPT2 tokens. Length: {len(tmp_dec_seq[0])}")
                            # for i in range(len(tmp_dec_seq[0])):
                            #     print(f"{i}: {tmp_dec_seq[0][i]} | {primary_tokenizer.decode(tmp_dec_seq[0][i])}", end=" ")
                            #     if i in indices[0]:
                            #         print(" -> located")
                            #     else:
                            #         print()
                            # print("-"*50) 
                    if ("beamsearch" in config['method']) or (config['method'] == "mlm-reranking"):
                        
                        candidate_total_losses, candidate_primary_losses, candidate_losses_for_loggings = score_hypotheses(source_batch,
                                                                                                                        hypotheses, 
                                                                                                                        config, 
                                                                                                                        lossfns,
                                                                                                                additional_batch=additional_batch, 
                                                                                                                context_batch=context_batch,
                                                                                                                use_context=use_context,
                                                                                                                label_ids=label_ids,
                                                                                                                keywords=keywords,
                                                                                                                kweight=new_kweight)
                        candidate_allsats = []
                        for losses_for_backward in candidate_losses_for_loggings:
                            allsat = True
                            for lossid, lossvalue in enumerate(losses_for_backward):
                                # if (lossid >= 1) and (losses_for_backward[lossid] > config['min_epsilons'][lossid - 1]):
                                if (lossid >= 1) and (losses_for_backward[lossid] > -np.log(config['min_epsilons'][lossid - 1])):
                                    allsat = False
                                    
                            candidate_allsats.append(allsat)
                        
                        if wandb.config.selection_criteria == "weighted_sum":
                            best_ix = np.argmin(np.array(candidate_total_losses))
                        elif wandb.config.selection_criteria == "allsat_primary":
                            allsat_ix = np.where(np.array(candidate_allsats)==True)[0]
                            if len(allsat_ix) > 0:
                                best_ix = np.argmin(np.array(candidate_primary_losses)[allsat_ix]) # select min primary loss among allsats
                                best_ix = allsat_ix[best_ix]
                            else: # if no candidate satisfying constraints, default to weighted_sum
                                best_ix = np.argmin(np.array(candidate_total_losses))
                    
                    
                    # elif config['method'] == "mlm-reranking":
                    #     ## define arguments that do not change across candidates
                    #     target_prefix = torch.empty((source_indices.size(0), 0)).long().to(config['device'])
                    #     primary_embed_dim = embed_luts[-1].embedding_dim
                    #     init = "target"
                    #     batch_size = 1
                    #     st = False
                    #     sampling_strategy = 'greedy'
                    #     sampling_strategy_k = 'none'
                    #     metric='l2'
                    #     same_embeds = True
                    #     final_bias = None
                    #     new_kweight = 5.0
                    #     step = 0
                    #     label_ids = [0, 0]
                    #     keywords = ["the" for _ in config['losses']]

                    #     candidate_total_losses = []
                    #     candidate_losses_for_loggings = []
                    #     candidate_allsats = []
                    #     candidate_primary_losses = []

                    #     with tqdm(total = len(hypotheses)) as pbar:
                    #         for ix in range(len(hypotheses)):

                    #             # logger.debug(f"== {ix} ==")
                    #             ## initialize embeddings
                    #             edit_candidate = hypotheses[ix]
                    #             sent_length = edit_candidate.size(1)
                    #             init_value = embed_luts[0](edit_candidate)
                    #             print(edit_candidate)
                    #             outputs = TargetEmbeddings(
                    #                         embed_dim=primary_embed_dim,
                    #                         embed_lut=embed_luts[0],
                    #                         sent_length=sent_length,
                    #                         batch_size=batch_size,
                    #                         device=config['device'],
                    #                         st=st,
                    #                         init_value=init_value, # initialize with current prediction
                    #                         random_init= init == "random",
                    #                         sampling_strategy=sampling_strategy,
                    #                         sampling_strategy_k=sampling_strategy_k,
                    #                         embed_scales=embed_scales,
                    #                         metric=metric,
                    #                         same_embed=same_embeds,
                    #                         final_bias=final_bias,
                    #                         eos_token_id=primary_tokenizer.eos_token_id
                    #                     )
                    #             pred_embeds, pred_tokens, pred_probs = outputs.forward_multiple(embed_luts, new_predictions=edit_candidate)
                                
                    #             ## c.f. What's happening inside. outputs.forward_multiple
                    #             # pred_tokens = hypotheses[ix]
                    #             # pred_probs = [None, None] # placeholder
                    #             # pred_embs = []
                    #             # for embed_lut in embed_luts:
                    #             #     pred_embs.append(embed_luts[0](hypotheses[ix]))
                    #             # pred_embeds = (pred_embs, embed_luts[0](hypotheses[ix]))

                    #             ## forward pass to calculate losses.
                    #             original_preds = None
                    #             if len(pred_embeds) > 1:
                    #                 original_preds = pred_embeds[1]

                    #             losses_for_backward = []

                    #             for lossid, lossname in enumerate(config['losses']):
                    #                 with torch.no_grad():
                    #                     lossvalue, logging_output =\
                    #                         lossfns[lossid].compute_loss(
                    #                             [source_batch, target_prefix], 
                    #                             [pred_tokens, pred_embeds[0][lossid], pred_probs], 
                    #                             additional_batch=None, 
                    #                             context_batch=None,
                    #                             use_context='false',
                    #                             embed_scale=embed_scales[lossid], 
                    #                             label_id=label_ids[lossid],
                    #                             keyword=keywords[lossid],
                    #                             original_preds=original_preds,
                    #                             kweight=new_kweight,
                    #                             step=step
                    #                         )

                    #                 losses_for_backward.append(lossvalue)  # for backward
                                    

                    #             ## calculate weighted sum of losses, check whether satisfying all constraints
                    #             allsat = True
                    #             total_loss = 0
                    #             losses_for_logging = []
                    #             for lossid, lossvalue in enumerate(losses_for_backward):
                    #                 if _lossid == 0:
                    #                     curr_loss += (1 - config['closs_weight']) * lossvalue.sum().item()
                    #                 else:
                    #                     curr_loss += config['closs_weight'] * lossvalue.sum().item()
                    #                 total_loss += curr_loss
                    #                 # total_loss += config['loss_weights'][lossid] * lossvalue.sum().item()
                    #                 losses_for_logging.append(lossvalue.sum().item())
                    #                 if (lossid == 0):
                    #                     candidate_primary_losses.append(lossvalue.sum().item())
                    #                 if (lossid >= 1) and (losses_for_backward[lossid] > config['min_epsilons'][lossid - 1]):
                    #                     allsat = False
                    #             candidate_total_losses.append(total_loss)
                    #             candidate_losses_for_loggings.append(losses_for_logging)
                    #             candidate_allsats.append(allsat)
                                
                    #             # step += 1 # not necessary, but doing it to avoid errors in model_wrapper.py forward
                    #             pbar.update(1)
                    #             for modelname in loss2modelname.values():
                    #                 name2model[modelname].zero_grad(set_to_none=True) 
                    #             torch.cuda.empty_cache()
                        
                    #     if config['selection_criteria'] == "weighted_sum":
                    #         best_ix = np.argmin(np.array(candidate_total_losses))
                    #     elif config['selection_criteria'] == "allsat_primary":
                    #         allsat_ix = np.where(np.array(candidate_allsats)==True)[0]
                    #         if len(allsat_ix) > 0:
                    #             best_ix = np.argmin(np.array(candidate_primary_losses)[allsat_ix]) # select min primary loss among allsats
                    #             best_ix = allsat_ix[best_ix]
                    #         else: # if no candidate satisfying constraints, default to weighted_sum
                    #             best_ix = np.argmin(np.array(candidate_total_losses))
                            
                    if _iter == 0: 
                        ## save the best prediction in a format compatible with mucola outputs
                        best_prediction = hypotheses[best_ix].squeeze(0).tolist()
                        # if config['method'] == "mlm-reranking":
                        #     predicted_batch = hypotheses[best_ix]
                        # else:
                        predicted_batch = hypotheses[best_ix].unsqueeze(0)
                        # logger.debug(best_prediction)
                        best_text = primary_tokenizer.decode(best_prediction)
                        # logger.debug(best_text)
                        best_allsat = candidate_allsats[best_ix]
                        best_losses = candidate_losses_for_loggings[best_ix]
                        best_weighted_loss = candidate_total_losses[best_ix]
                        
                        
                        # logger.debug(f"best_prediction: {best_prediction}")
                        logger.debug(f"best_text: {best_text}")
                        # logger.debug(f"best_allsat: {best_allsat}")
                        # logger.debug(f"best_losses: {best_losses}")
                        # logger.debug(f"best_weighted_loss: {best_weighted_loss}")
                    else:
                        update = False
                        if wandb.config.selection_criteria == "weighted_sum":
                            if best_weighted_loss > candidate_total_losses[best_ix]:
                                update = True
                        elif wandb.config.selection_criteria == "allsat_primary":
                            if best_allsat == False and candidate_allsats[best_ix] == True:
                                update = True
                            elif best_allsat == False and candidate_allsats[best_ix] == False:
                                if best_weighted_loss > candidate_total_losses[best_ix]:
                                    update = True
                            elif best_allsat == True and candidate_allsats[best_ix] == True:
                                if best_losses[0] > candidate_losses_for_loggings[best_ix][0]:
                                    update = True
                        if update:
                            ## save the best prediction in a format compatible with mucola outputs
                            best_prediction = hypotheses[best_ix].squeeze(0).tolist()
                            # if config['method'] == "mlm-reranking":
                            #     predicted_batch = hypotheses[best_ix]
                            # else:
                            predicted_batch = hypotheses[best_ix].unsqueeze(0)
                            # logger.debug(best_prediction)
                            best_text = primary_tokenizer.decode(best_prediction)
                            # logger.debug(best_text)
                            best_allsat = candidate_allsats[best_ix]
                            best_losses = candidate_losses_for_loggings[best_ix]
                            best_weighted_loss = candidate_total_losses[best_ix]
                            
                            logger.debug(f"iter {_iter}. Update best prediction")
                            # logger.debug(f"best_prediction: {best_prediction}")
                            logger.debug(f"best_text: {best_text}")
                            # logger.debug(f"best_allsat: {best_allsat}")
                            # logger.debug(f"best_losses: {best_losses}")
                            # logger.debug(f"best_weighted_loss: {best_weighted_loss}")
                        
                        if best_allsat:
                            es_patience_count += 1
                            if config['early_stopping_patience'] == -1:
                                continue
                            if es_patience_count > config['early_stopping_patience']:
                                logger.info(f"early stopping at iter {_iter}")
                                break
                
                if sample_idx == 0:
                    output = {
                        "prompt":{
                            "text":source_text,
                            "tokens":source_indices.tolist()
                            }, 
                        "generations":[{
                            "text": best_text,
                            "tokens": best_prediction,
                            "indices": indices, 
                            "orig_tokens_at_indices": name2tokenizer[config['model_paths'][0]].decode(original_sequence[:, indices].squeeze()),
                            "allsat": best_allsat,
                            "losses": best_losses,
                            "weighted_loss": best_weighted_loss
                            }]
                    }
                else:
                    output['generations'].append(
                        {
                            "text": best_text,
                            "tokens": best_prediction,
                            "indices": indices, 
                            "orig_tokens_at_indices": name2tokenizer[config['model_paths'][0]].decode(original_sequence[:, indices].squeeze()),
                            "allsat": best_allsat,
                            "losses": best_losses,
                            "weighted_loss": best_weighted_loss
                        }
                    )
            
                if sample_idx + 1 == config['num_samples']:
                    json.dump(output, outf)
                    outf.write("\n")
                    outf.flush()

        
    outf.close()

    
    if config['resume']:
        run.summary["decode_time"] += (time.time() - decode_start_time)
    else:
        run.summary["decode_time"] = (time.time() - decode_start_time)
    run.summary["num_skipped"] = num_skipped
    run.summary["num_edited"] = num_edited
    
    run.finish()
    
    if config['task'] == 'toxicity':
        # evaluate(run.path, outfile, 'toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n')
        evaluate(run.path, outfile, 'toxicity-int,ppl-big,dist-n,repetition,fluency',
                 toxicity_model_path=config['model_paths'][1], toxicity_model_type=config['model_types'][1]) # 시간 문제로, perspective api 제외
    elif config['task'] == 'formality':
        evaluate(run.path, outfile, 'formality-int,formality-ext,ppl-big,dist-n,repetition,fluency', 
                 formality_model_path=config['model_paths'][1],formality_model_type=config['model_types'][1])
    elif config['task'] == 'sentiment':
        evaluate(run.path, outfile, 'sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency',
                 sentiment_model_path=config['model_paths'][1],sentiment_model_type=config['model_types'][1])
    elif config['task'] == 'sentiment-lewis-compr':
        evaluate(run.path, outfile, 'sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency',
                 sentiment_model_path=config['model_paths'][1],sentiment_model_type=config['model_types'][1])
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Locally Editing Text Generation')
    parser.add_argument('--task', type=str, help='task name', choices=['toxicity', 'formality', 'sentiment', 'sentiment-lewis-compr'])
    parser.add_argument('--source_data', type=str, default='data/formality/GYAFC_Corpus/Entertainment_Music/test/informal', help='source data path')
    parser.add_argument('--source_style', type=str, default="informal", help='source style')
    parser.add_argument('--target_style', type=str, default="formal", help='target style')
    parser.add_argument('--target_label_ids', nargs='+', type=int, default=[1,1], help='a list of indices of target label used in each of models. e.g. [1,1]')
    parser.add_argument('--model_paths', nargs='+', type=str, default=['gpt2-large', '/home/s3/hyeryung/data/loc_edit/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17'], help='model paths')
    parser.add_argument('--tokenizer_paths', nargs='+', type=str, default=['gpt2-large', '/home/s3/hyeryung/data/loc_edit/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17'], help='tokenizer paths')
    parser.add_argument('--model_types', nargs='+', type=str, default=['AutoModelForCausalLM', 'RobertaCustomForSequenceClassification'], help='model types')
    parser.add_argument('--output_dir_prefix', type=str, help='output directory prefix. e.g. outputs/formality/mlm-reranking')
    parser.add_argument('--early_stopping_patience', type=int, default=-1, help='early stopping patience')
    parser.add_argument('--method', type=str, default="mlm-beamsearch-v0", help='method name', choices=['mlm-beamsearch-v0', 'mlm-beamsearch-v1', 'mlm-beamsearch-v2', 'mlm-reranking'])
    parser.add_argument('--locate_unit', type=str, default="token", help='unit to locate')
    parser.add_argument('--min_epsilons', nargs='+', type=float, default=[0.75], help='min epsilons')
    parser.add_argument('--num_samples', type=int, default=1, help='number of samples to edit per prompt')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--target_type', type=str, default='embeds', help="target type (embeds, simplex, probability) from prior work's code")
    parser.add_argument('--cache_dir', type=str, default='hf_cache', help='cache directory')
    parser.add_argument('--jsonl_primary_key', type=str, default="prompt", help='jsonl primary key')
    parser.add_argument('--jsonl_secondary_key', type=str, default="text", help='jsonl secondary key')
    parser.add_argument('--losses', nargs='+', type=str, default=['gpt2', 'classification_no_prefix'], help='losses')
    parser.add_argument('--build_loss_dict', type=json.loads, default='{"coeff_steps": 200, "coeff_pattern": "constant", "loss_type": "xentropy", "length_normalize": false, "AR_temperature": 1.0, "AR_top_k": 0, "AR_top_p": 0.96, "max_output_length": 20}', help='build loss dict')
    parser.add_argument('--num_edit_token_per_step', type=int, default=5, help='number of edit tokens per step')
    parser.add_argument('--k_per_location', type=int, default=15, help='k per location')
    parser.add_argument('--n_iter', type=int, default=3, help='number of iterations')
    parser.add_argument('--selection_criteria', type=str, default="weighted_sum", help='selection criteria')
    parser.add_argument('--closs_weight', type=float, default=0.32, help='closs weight')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size')   
    parser.add_argument('--wandb_project', type=str, default='mlm_reranking', help='wandb project name')   
    parser.add_argument('--wandb_entity', type=str, default='hayleyson', help='wandb entity name')
    parser.add_argument('--wandb_run_id', type=str, help='wandb run name')
    parser.add_argument('--resume', action='store_true', help='whether to resume from a previous run')
    parser.add_argument('--slurm_job_id',  type=str, help='slurm job id (for debugging)')
 

    args = parser.parse_args()
    config = vars(args)
        
    
    main(config)
    
    