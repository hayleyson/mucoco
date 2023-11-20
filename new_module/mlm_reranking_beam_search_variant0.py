#!/usr/bin/env python
# coding: utf-8


# standard libraries
import os
import sys
import json
import logging
from collections import namedtuple
sys.path.append("/home/s3/hyeryung/mucoco")
os.chdir("/home/s3/hyeryung/mucoco")

# installed packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
import wandb

# custom libraries
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, OptimizerLE, get_epsilon, locate
from new_module.evaluate_wandb import main
from new_module.utils import score_hypotheses, constrained_beam_search_v0

## hyperparemeters
config = dict(
early_stopping_patience=-1,
method='mlm-beamsearch',
selection_criteria = "weighted_sum", #"allsat_primary",
num_edit_token_per_step = 4,
locate_unit = "token",
k_per_location = 10,
loss_weights = [0.5, 0.5],
min_epsilons = [-3],
n_iter = 2,
num_samples = 10,
device = "cuda",
target_type='embeds',
cache_dir='hf_cache',
jsonl_primary_key = "prompt",
jsonl_secondary_key = "text",
losses = ['gpt2', 'classification_no_prefix'],
build_loss_dict={'coeff_steps': 200, 
                'coeff_pattern': 'constant',
                'loss_type': 'xentropy',#'dotplusplus',
                'length_normalize': False,
                'AR_temperature': 1.0,
                'AR_top_k': 0,
                'AR_top_p': 0.96,
                'max_output_length': 20},
model_paths=['gpt2-large',
            'models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best'],
tokenizer_paths=['gpt2-large',
                'models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best'],
model_types=['AutoModelForCausalLM',
            'RobertaCustomForSequenceClassification'],
source_data = 'new_module/toxicity-avoidance/data/testset_gpt2_2500.jsonl',
beam_size = 3
)

display_name = f"test-{config['method']}-{config['locate_unit']}-nps{config['num_edit_token_per_step']}-k{config['k_per_location']}-beam{config['beam_size']}-{config['selection_criteria']}-{config['loss_weights'][0]}-{config['loss_weights'][1]}-wandb"
run = wandb.init(project="mucola", config=config, name=display_name)

class dummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
build_loss_args=dummyArgs(**config['build_loss_dict'])

## logging-related
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("le")
logger.setLevel(os.environ.get('LOGGING_LEVEL', logging.DEBUG))
logger.setLevel(logging.INFO)

outdir = os.path.join('outputs/toxicity/mlm-reranking', display_name
                      )
os.makedirs(outdir, exist_ok=True)
outfile = f"{outdir}/outputs_epsilon{config['min_epsilons'][0]}.txt"
outf = open(outfile, "w")




## load data
source_dataset = [json.loads(l)[config['jsonl_primary_key']][config['jsonl_secondary_key']] for l in open(config['source_data'])]
generation_dataset = [json.loads(l)["generations"] for l in open(config['source_data'])]
# config['source_data = 'new_module/toxicity-avoidance/data/testset_jigsaw_1960.jsonl'
# source_dataset = ["" for l in open(config['source_data'])]
# generation_dataset = [json.loads(l)["source"] for l in open(config['source_data'])]

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

for i, model_path in enumerate(config['model_paths']):
    if model_path not in name2model: #making sure we are not loading the model twice in case some constraints use the same model. 
        name2tokenizer[model_path] = AutoTokenizer.from_pretrained(config['tokenizer_paths'][i], cache_dir=config['cache_dir'],  use_fast=True)
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

## for editing
model_checkpoint = "roberta-base"
mlm_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
mlm = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

primary_tokenizer.add_special_tokens({'mask_token':mlm_tokenizer.mask_token})
primary_mask_token_id = primary_tokenizer.mask_token_id


## ------------------------- beginning of main logic ------------------------- ##
text_id = 0
for text_id in range(len(source_dataset)):
    source_text = source_dataset[text_id]
    source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(config['device'])
    source_batch = torch.cat([source_indices], dim=0).to(config['device'])
    
    predicted_batches = [x["tokens"] for x in generation_dataset[text_id]]
    predicted_batches = [torch.tensor([x], dtype=torch.long, device=config['device']) for x in predicted_batches]
    AR_prediction_all = [x["text"] for x in generation_dataset[text_id]]
    
    sample_idx = 0
    for sample_idx in range(config['num_samples'])[:]:
        predicted_batch = predicted_batches[sample_idx].cuda()
        AR_prediction = AR_prediction_all[sample_idx]
        
        logger.critical(f"text_id {text_id} sample_id {sample_idx}")# \n[text] {AR_prediction} \n[input_ids] {predicted_batch}")
        # logger.critical(AR_prediction)
        # logger.critical(AR_prediction)
        
        ## check if toxicity less than threshold
        gold_losses = []
        label_ids = [0, 0]
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
                        kweight=new_kweight
                    )
            gold_losses.append(lossvalue.squeeze().item())
            if (lossid >= 1) and (gold_losses[lossid] > config['min_epsilons'][lossid - 1]):
                allsat = False
        
        if allsat:
            logger.info(f"skipping this sample since it already satisfies constraint. {gold_losses}")
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
                
            es_patience_count = 0
            best_ix, best_prediction, best_text, best_allsat, best_losses, best_weighted_loss = None, None, None, None, None, None
            
            _iter = 0
            for _iter in range(config['n_iter']):
                
                ## locate tokens to edit
                batch = {"input_ids": predicted_batch}
                indices = locate(name2model[config['model_paths'][1]], 
                                name2tokenizer[config['model_paths'][1]], 
                                batch, 
                                max_num_tokens=config['num_edit_token_per_step'], 
                                unit=config['locate_unit'], 
                                use_cuda=True)
                logger.debug(f"iter {_iter}, sample_idx: {sample_idx}")
                logger.debug(f"located indices: {indices}")
                logger.debug(f"located indices: {name2tokenizer[config['model_paths'][1]].decode(predicted_batch[:, indices].squeeze())}")
                # logger.debug(f"located indices: {predicted_batch[:, indices]}")
                
                ## replace tokens at the indices with mask tokens
                masked_sequence = predicted_batch.clone().detach()
                for i in indices:
                    masked_sequence[:, i] = primary_mask_token_id
                masked_sequence_text = primary_tokenizer.batch_decode(masked_sequence.tolist())
                inputs = mlm_tokenizer(masked_sequence_text, return_tensors="pt")
                
                # ## c.f. check if spaces are preserved. -> preserved! checked.
                # logger.debug(inputs['input_ids'])
                # logger.debug(mlm_tokenizer.decode(inputs['input_ids'][0]))
                
                ## make predictions for the masked indices
                with torch.no_grad():
                    logits = mlm(**inputs).logits
                indices_in_mlm_tokens = (inputs.input_ids == mlm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                
                ## get top k tokens for each index 
                predicted_token_ids = torch.topk(logits[0, indices_in_mlm_tokens], k=config['k_per_location'], dim=-1)
                # logger.debug(predicted_token_ids) # shape : (config['num_edit_token_per_step'],  config['k_per_location'])
                
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

                # logger.debug(f"masked_sequence: {masked_sequence} \n mask_token_index: {mask_token_index} \n \
                #                primary_mask_token_id: {primary_mask_token_id} \n predicted_token_ids: {predicted_token_ids}")
                # logger.debug(f"additional_batch={additional_batch}, source_batch = {source_batch}, context_batch={context_batch},\
                # use_context={use_context},label_id={label_ids[lossid]},keyword={keywords[lossid]},kweight={new_kweight}")
                hypotheses = constrained_beam_search_v0(source_batch,
                                                        masked_sequence,
                                                        torch.LongTensor(sorted(indices[0])), 
                                                        primary_mask_token_id, 
                                                        predicted_token_ids, 
                                                        primary_tokenizer, 
                                                        mlm_tokenizer,
                                                        lossfns,
                                                        config, 
                                                        beam_size = config['beam_size'],
                                                        additional_batch=additional_batch, 
                                                        context_batch=context_batch,
                                                        use_context=use_context,
                                                        label_ids=label_ids,
                                                        keywords=keywords,
                                                        kweight=new_kweight
                                                        )
        
        
                
        
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
                        if (lossid >= 1) and (losses_for_backward[lossid] > config['min_epsilons'][lossid - 1]):
                            allsat = False
                    candidate_allsats.append(allsat)
                
                if config['selection_criteria'] == "weighted_sum":
                    best_ix = np.argmin(np.array(candidate_total_losses))
                elif config['selection_criteria'] == "allsat_primary":
                    allsat_ix = np.where(np.array(candidate_allsats)==True)[0]
                    if len(allsat_ix) > 0:
                        best_ix = np.argmin(np.array(candidate_primary_losses)[allsat_ix]) # select min primary loss among allsats
                        best_ix = allsat_ix[best_ix]
                    else: # if no candidate satisfying constraints, default to weighted_sum
                        best_ix = np.argmin(np.array(candidate_total_losses))
                    
                if _iter == 0: 
                    ## save the best prediction in a format compatible with mucola outputs
                    best_prediction = hypotheses[best_ix].squeeze().tolist()
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
                    if config['selection_criteria'] == "weighted_sum":
                        if best_weighted_loss > candidate_total_losses[best_ix]:
                            update = True
                    elif config['selection_criteria'] == "allsat_primary":
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
                        best_prediction = hypotheses[best_ix].squeeze().tolist()
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

main(run.path, outfile, 'toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n')

