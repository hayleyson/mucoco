import os
import sys
sys.path.append("/home/s3/hyeryung/mucoco")
import logging

from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
import wandb
import torch.nn.functional as F

import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, OptimizerLE, get_epsilon, locate
from new_module.evaluate_wandb import main

## hyperparemeters
config = dict(
early_stopping_patience=-1,
method='mlm',
selection_criteria = "weighted_sum", #"allsat_primary",
num_edit_token_per_step = 4,
locate_unit = "token",
k_per_location = 3,
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
                'loss_type': 'dotplusplus',
                'length_normalize': False,
                'AR_temperature': 1.0,
                'AR_top_k': 0,
                'AR_top_p': 0.96,
                'max_output_length': 20},
model_paths=['gpt2-large',
            'models/models_mucola/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best'],
tokenizer_paths=['gpt2-large',
                'models/models_mucola/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best'],
model_types=['AutoModelForCausalLM',
            'RobertaCustomForSequenceClassification'],
source_data = 'new_module/toxicity-avoidance/data/testset_gpt2_2500.jsonl'
)

display_name = f"tmp-mlm-{config['locate_unit']}-nps{config['num_edit_token_per_step']}-k{config['k_per_location']}-{config['selection_criteria']}-{config['loss_weights'][0]}-{config['loss_weights'][1]}-wandb"
run = wandb.init(project="mucola", config=config, name=display_name)

class dummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
build_loss_args=dummyArgs(**config['build_loss_dict'])

## logging-related
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("le")
logger.setLevel(logging.INFO)

outdir = os.path.join('/home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking', display_name
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
    for sample_idx in range(config['num_samples']):
        predicted_batch = predicted_batches[sample_idx].cuda()
        AR_prediction = AR_prediction_all[sample_idx]

        logger.critical("Original output")
        logger.critical(AR_prediction)
        
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
                mask_token_index = (inputs.input_ids == mlm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

                ## get top k tokens for each index 
                predicted_token_ids = torch.topk(logits[0, mask_token_index], k=config['k_per_location'], dim=-1)
                # logger.debug(predicted_token_ids) # shape : (config['num_edit_token_per_step'],  config['k_per_location'])

                ## get k ** num_located_indices sequences with different combinations of the top k tokens for located locations
                ## test_sequences will hold a list of input ids encoded with GPT2Tokenizer
                test_sequences = []
                num_located_tokens = len(indices[0])
                num_all_cases = config['k_per_location'] ** num_located_tokens
                tok_cand_combo = [0 for i in range(num_located_tokens)]
                for case_id in range(num_all_cases):
                    for i in range(num_located_tokens):
                        tok_cand_combo[i] = (case_id // (config['k_per_location']**i)) % config['k_per_location']
                    
                    tmp_seq = inputs['input_ids'].clone()
                    for pos_id, tok_cand_id in enumerate(tok_cand_combo):
                        tmp_seq[0, mask_token_index[pos_id]] = predicted_token_ids.indices[pos_id, tok_cand_id]
                    
                    # need to do decode with RobertaTokenizer and encode with GPT2Tokenizer
                    # logger.debug(mlm_tokenizer.batch_decode(tmp_seq[:, mask_token_index], skip_special_tokens=True))
                    tmp_dec_seq = primary_tokenizer(mlm_tokenizer.batch_decode(tmp_seq, skip_special_tokens=True), return_tensors="pt").input_ids.cuda()
                    test_sequences.append(tmp_dec_seq)

                ## pass the candidates through loss functions and calculate weighted sum of losses.

                ## define arguments that do not change across candidates
                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(config['device'])
                primary_embed_dim = embed_luts[-1].embedding_dim
                init = "target"
                batch_size = 1
                st = False
                sampling_strategy = 'greedy'
                sampling_strategy_k = 'none'
                metric='l2'
                same_embeds = True
                final_bias = None
                new_kweight = 5.0
                step = 0
                label_ids = [0, 0]
                keywords = ["the" for _ in config['losses']]

                candidate_total_losses = []
                candidate_losses_for_loggings = []
                candidate_allsats = []
                candidate_primary_losses = []

                with tqdm(total = len(test_sequences)) as pbar:
                    for ix in range(len(test_sequences)):

                        # logger.debug(f"== {ix} ==")
                        ## initialize embeddings
                        edit_candidate = test_sequences[ix]
                        sent_length = edit_candidate.size(1)
                        init_value = embed_luts[0](edit_candidate)
                        outputs = TargetEmbeddings(
                                    embed_dim=primary_embed_dim,
                                    embed_lut=embed_luts[0],
                                    sent_length=sent_length,
                                    batch_size=batch_size,
                                    device=config['device'],
                                    st=st,
                                    init_value=init_value, # initialize with current prediction
                                    random_init= init == "random",
                                    sampling_strategy=sampling_strategy,
                                    sampling_strategy_k=sampling_strategy_k,
                                    embed_scales=embed_scales,
                                    metric=metric,
                                    same_embed=same_embeds,
                                    final_bias=final_bias,
                                    eos_token_id=primary_tokenizer.eos_token_id
                                )
                        pred_embeds, pred_tokens, pred_probs = outputs.forward_multiple(embed_luts, new_predictions=edit_candidate)
                        
                        ## c.f. What's happening inside. outputs.forward_multiple
                        # pred_tokens = test_sequences[ix]
                        # pred_probs = [None, None] # placeholder
                        # pred_embs = []
                        # for embed_lut in embed_luts:
                        #     pred_embs.append(embed_luts[0](test_sequences[ix]))
                        # pred_embeds = (pred_embs, embed_luts[0](test_sequences[ix]))

                        ## forward pass to calculate losses.
                        original_preds = None
                        if len(pred_embeds) > 1:
                            original_preds = pred_embeds[1]

                        losses_for_backward = []

                        for lossid, lossname in enumerate(config['losses']):
                            with torch.no_grad():
                                lossvalue, logging_output =\
                                    lossfns[lossid].compute_loss(
                                        [source_batch, target_prefix], 
                                        [pred_tokens, pred_embeds[0][lossid], pred_probs], 
                                        additional_batch=None, 
                                        context_batch=None,
                                        use_context='false',
                                        embed_scale=embed_scales[lossid], 
                                        label_id=label_ids[lossid],
                                        keyword=keywords[lossid],
                                        original_preds=original_preds,
                                        kweight=new_kweight,
                                        step=step
                                    )

                            losses_for_backward.append(lossvalue)  # for backward
                            

                        ## calculate weighted sum of losses, check whether satisfying all constraints
                        allsat = True
                        total_loss = 0
                        losses_for_logging = []
                        for lossid, lossvalue in enumerate(losses_for_backward):
                            total_loss += config['loss_weights'][lossid] * lossvalue.sum().item()
                            losses_for_logging.append(lossvalue.sum().item())
                            if (lossid == 0):
                                candidate_primary_losses.append(lossvalue.sum().item())
                            if (lossid >= 1) and (losses_for_backward[lossid] > config['min_epsilons'][lossid - 1]):
                                allsat = False
                        candidate_total_losses.append(total_loss)
                        candidate_losses_for_loggings.append(losses_for_logging)
                        candidate_allsats.append(allsat)
                        
                        # step += 1 # not necessary, but doing it to avoid errors in model_wrapper.py forward
                        pbar.update(1)
                        for modelname in loss2modelname.values():
                            name2model[modelname].zero_grad(set_to_none=True) 
                        torch.cuda.empty_cache()
                
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
                    best_prediction = test_sequences[best_ix].squeeze().tolist()
                    predicted_batch = test_sequences[best_ix]
                    # logger.debug(best_prediction)
                    best_text = primary_tokenizer.decode(best_prediction)
                    # logger.debug(best_text)
                    best_allsat = candidate_allsats[best_ix]
                    best_losses = candidate_losses_for_loggings[best_ix]
                    best_weighted_loss = candidate_total_losses[best_ix]
                    
                    
                    logger.debug(f"best_prediction: {best_prediction}")
                    logger.debug(f"best_text: {best_text}")
                    logger.debug(f"best_allsat: {best_allsat}")
                    logger.debug(f"best_losses: {best_losses}")
                    logger.debug(f"best_weighted_loss: {best_weighted_loss}")
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
                        best_prediction = test_sequences[best_ix].squeeze().tolist()
                        predicted_batch = test_sequences[best_ix]
                        # logger.debug(best_prediction)
                        best_text = primary_tokenizer.decode(best_prediction)
                        # logger.debug(best_text)
                        best_allsat = candidate_allsats[best_ix]
                        best_losses = candidate_losses_for_loggings[best_ix]
                        best_weighted_loss = candidate_total_losses[best_ix]
                        
                        logger.debug(f"iter {_iter}. Update best prediction")
                        logger.debug(f"best_prediction: {best_prediction}")
                        logger.debug(f"best_text: {best_text}")
                        logger.debug(f"best_allsat: {best_allsat}")
                        logger.debug(f"best_losses: {best_losses}")
                        logger.debug(f"best_weighted_loss: {best_weighted_loss}")
                    
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