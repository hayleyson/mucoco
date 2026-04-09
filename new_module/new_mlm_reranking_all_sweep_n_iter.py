#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from itertools import chain
import functools
import math
import argparse
import json
import logging
import os
import time
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

import new_module.losses as lossbuilder
import wandb
# from new_module.decode_utils import (
#     beam_rerank_v0,
#     beam_rerank_v1,
#     beam_rerank_v2,
#     combi_rerank,
# )
from new_module.new_decode_utils import get_beam_hypotheses_v0, get_beam_hypotheses_v1, get_combi_hypotheses, final_reranking, analyze_span_lengths_and_count, editing_with_delete_variable_replace
from new_module.evaluation.evaluate_pipeline import run_generation_evaluation
from new_module.locate.new_locate_utils import LocateMachine
from new_module.utils.robertacustom import RobertaCustomForSequenceClassification
from new_module.em_training.nli.models import EncoderModel

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGGING_LEVEL", logging.DEBUG))


def main(config):
    
    main_start_time = time.time()

    if not config.get("model_tag", None):
        if ("energy-training" in config["model_paths"][1]) or ("finegrained_labels" in config["model_paths"][1]): 
            config["model_tag"] = "em"
        else:
            config["model_tag"] = "clsf"

    # used fixed values for build_loss_dict
    config["build_loss_dict"] = {"length_normalize": True, 
                                 "alpha": 1.0, 
                                 "AR_temperature": 1.0, # unused
                                 "AR_top_k": 0, # unused
                                 "AR_top_p": 0.96, # unused
                                 "max_output_length": 20 # unused
                                 }
    
    if config["resume"]:
        logger.info("resuming from a previous run")
        run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            id=config["wandb_run_id"],
            resume="must",
        )
    else:
        run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            config=config,
        )
    
    config["k_per_location"] = wandb.config.k_per_location
    config["beam_size"] = wandb.config.beam_size
    
    run_id = run.path.split("/")[-1]
    display_name = f"{run_id}"
    
    outdir = os.path.join(config["output_dir_prefix"], display_name)
    os.makedirs(outdir, exist_ok=True)
    outfile = f"{outdir}/outputs_epsilon{config['min_epsilons'][0]}.txt"
    run.summary["outfile_path"] = outfile


    ## load data
    if (config["task"] == "toxicity") or (config["task"] == "sentiment") or (config["task"] == "nli"):
        source_dataset = [
            json.loads(l)[config["jsonl_primary_key"]][config["jsonl_secondary_key"]]
            for l in open(config["source_data"])
        ]
        generation_dataset = [
            json.loads(l)["generations"] for l in open(config["source_data"])
        ]
    elif (config["task"] == "formality"):
        with open(config["source_data"], "r") as f:
            generation_dataset = [line.rstrip('\n') for line in f.readlines()]
        source_dataset = ["" for l in generation_dataset]

    # check if outfile exists
    if (config["resume"]) and (os.path.exists(outfile)):

        raise NotImplementedError

        # with open(outfile, "r") as f:
        #     existing_gens = [x.rstrip("\n") for x in f.readlines()]
        # resume_idx = len(existing_gens)
        # if resume_idx == len(source_dataset):
        #     logger.debug("output file is already complete. skipping this run.")
        #     return
        # elif resume_idx < len(source_dataset):
        #     logger.info(
        #         f"output file already exists but is incomplete. resuming from index: {resume_idx}"
        #     )
        #     outf = open(outfile, "a")
        #     int_outf = open(outfile+".intermediate", "a")
        # else:
        #     logger.critical(
        #         f"output file seems to be corrupted. The file length is {resume_idx}, where the size of source_dataset is {len(source_dataset)}"
        #     )
        #     return
    else:
        resume_idx = 0
        # outf = open(outfile, "w")
        # int_outf = open(outfile+".intermediate", "w")
        outfs= dict()
        for _iter in range(config['n_iter']):
            outfs[_iter] = open(outfile+f".{_iter}", "w")

    ## load tokenizer, models, define losses
    name2tokenizer = {}
    name2model = {}
    name2config = {}
    loss2tokenizer = {}

    for i, model_path in enumerate(config["model_paths"]):
        if (
            model_path not in name2model
        ):  # making sure we are not loading the model twice in case some constraints use the same model.
            
            if config["model_types"][i] == "EncoderModel":
                # config
                with open(os.path.join(config["model_paths"][i], 'config.json')) as f:
                    model_config = json.load(f)
                model_config['device'] = config['device']
                model_config['model_path'] = os.path.join(config["model_paths"][i], 'best_model_pearsonr.pth')
                if config["locate_method"] == "attention":
                    model_config['locate']['type'] = "attention"
                elif config["locate_method"] == "grad_norm":
                    model_config['locate']['type'] = "gradnorm"
                name2config[model_path] = model_config
                
                # load model
                model = EncoderModel(params=name2config[model_path])
                model.load_state_dict(torch.load(name2config[model_path]["model_path"],weights_only=True),strict=False)
                name2model[model_path] = lossbuilder.ModelWrapper(model)
                name2model[model_path].eval()
                name2model[model_path].to(config['device'])
                
                # tokenizer
                name2tokenizer[config["tokenizer_paths"][i]] = name2model[model_path].tokenizer
                
            else:   
                name2config[model_path] = AutoConfig.from_pretrained(
                    model_path, cache_dir=config["cache_dir"]
                )

                if config["model_types"][i] == "RobertaCustomForSequenceClassification":
                    name2model[model_path] = lossbuilder.ModelWrapper(
                        RobertaCustomForSequenceClassification.from_pretrained(
                            model_path,
                            config=name2config[model_path],
                            cache_dir=config["cache_dir"],
                        )
                    )
                    
                else:
                    name2model[model_path] = lossbuilder.ModelWrapper(
                        getattr(transformers, config["model_types"][i]).from_pretrained(
                            model_path,
                            config=name2config[model_path],
                            cache_dir=config["cache_dir"],
                        )
                    )
                name2model[model_path].eval()
                name2model[model_path].to(config['device'])
            
                try:
                    name2tokenizer[config["tokenizer_paths"][i]] = AutoTokenizer.from_pretrained(
                        config["tokenizer_paths"][i],
                        cache_dir=config["cache_dir"],
                        use_fast=True,
                    )
                except:
                    name2tokenizer[config["tokenizer_paths"][i]] = AutoTokenizer.from_pretrained(
                        config["tokenizer_paths"][i],
                        cache_dir=config["cache_dir"],
                        use_fast=False,
                    )

    # for faster experiment. from internal ablation, performance didn't degrade much
    name2model[config["model_paths"][0]].half()

    mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    mlm = None if config["method"] == "mlm-beamsearch-v2" else AutoModelForMaskedLM.from_pretrained("roberta-base").to(config['device'])

    class dummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    build_loss_args = dummyArgs(**config["build_loss_dict"])
    build_loss_args.task = config["task"]

    lossfns = []
    for i, loss in enumerate(config["losses"]):
        lossfns.append(
            lossbuilder.build_loss(
                loss,
                name2model[config["model_paths"][i]],
                name2tokenizer[config["tokenizer_paths"][i]],
                build_loss_args,
            )
        )
        lossfns[i].tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})
        loss2tokenizer[loss] = lossfns[i].tokenizer

    # define an object to locate problematic phrases
    locator = LocateMachine(lossfns[1].model, lossfns[1].tokenizer, config['task'])

    if getattr(wandb.config, "closs_weight", None) is not None: ## closs_weight is used if sweep is used
        config["loss_weights"] = [1, wandb.config.closs_weight]
        run.config.update({"closs_weight": config["loss_weights"]}, allow_val_change=True)
    logger.info(f"loss_weights: {config['loss_weights']}")


    run.summary["prep_time"] = time.time() - main_start_time
    ## beginning of main logic
    decode_start_time = time.time()
    
    if config["resume"]:
        num_skipped = run.summary.get("num_skipped", 0)
        num_edited = run.summary.get("num_edited", 0)
        num_decoded_tokens = run.summary.get("num_decoded_tokens", 0)
    else:
        num_skipped = 0
        num_edited = 0
        num_decoded_tokens = 0

    interrupted = False
    if (config["task"] == "toxicity") or (config["task"] == "sentiment") or (config["task"] == "nli"):
        text_id_interval = 1
    elif (config["task"] == "formality"):
        text_id_interval = config['num_samples']
        
    iteration_specific_execution_times = {i: 0 for i in range(config['n_iter'])}
        
    for text_id in range(resume_idx, len(source_dataset), text_id_interval):
        source_text = source_dataset[text_id]
        if (source_text == "") and (lossfns[0].tokenizer.bos_token is not None):
            source_text = lossfns[0].tokenizer.bos_token
        elif (source_text == "") and (lossfns[0].tokenizer.bos_token is None):
            source_text = " "

        if (config["task"] == "toxicity") or (config["task"] == "sentiment") or (config["task"] == "nli"):
            AR_prediction_all = [x["text"] for x in generation_dataset[text_id]]
            # predicted_batches = [x["tokens"] for x in generation_dataset[text_id]]
            # predicted_batches = [
            #     torch.tensor([x], dtype=torch.long, device=config["device"])
            #     for x in predicted_batches
            # ]
            
        elif (config["task"] == "formality"):
            # AR_prediction_all = [generation_dataset[text_id]]
            AR_prediction_all = generation_dataset[text_id: text_id + text_id_interval]
 
        curr_num_samples = len(AR_prediction_all)
        if curr_num_samples == 0:
            continue

        # --------------------------------------------------------------------------------------------- #
        ## check whether initial text satisfies constraint

        curr_loss = torch.zeros(len(AR_prediction_all)).to(config['device'])
        logging_loss = torch.zeros((len(AR_prediction_all),len(config["losses"]))).to(config['device'])
        edit_yn = torch.ones(len(AR_prediction_all), dtype=torch.bool).to(config['device'])
                
        for lossid, lossname in enumerate(config["losses"]):
            with torch.no_grad():
                lossvalue = lossfns[lossid].compute_gold_loss(
                    source_text, AR_prediction_all,
                    label_id=config['target_label_ids'][lossid],
                )
                torch.cuda.empty_cache()
            curr_loss += config["loss_weights"][lossid] * lossvalue
            logging_loss[:, lossid] = lossvalue.clone()


        allsat = logging_loss[:,1] < -math.log(config["min_epsilons"][0])
        allsat_ix = allsat.nonzero().squeeze(0)
        if (not config["dont_skip_allsat"]):
            edit_yn[allsat_ix] = False
        edited_at_all_yn = edit_yn.detach().clone()
        
        es_patience_count = torch.zeros(len(AR_prediction_all),dtype=torch.long).to(config['device'])
        best_allsat = allsat.detach().clone()
        best_losses = logging_loss.detach().clone()
        best_weighted_loss = curr_loss.detach().clone()            
        best_text = deepcopy(AR_prediction_all)
        running_text = [x for i, x in enumerate(AR_prediction_all) if edit_yn[i]] ## hold only samples that need be edited
        int_output = [{} for _ in range(len(AR_prediction_all))]

        if (edit_yn.sum().item() == 0) and (not config["dont_skip_allsat"]):
            
            num_edited += 0
            num_skipped += len(AR_prediction_all)
            num_decoded_tokens += 0
            
            
            logger.info(
                    f"skipping this sample since it already satisfies constraint. {best_losses}"
                )
            # save data as is
            for _iter in range(config['n_iter']):
                output = {
                            "prompt": {
                                "text": source_text,
                            },
                            "generations": [
                                {
                                    "text": best_text[i],
                                    "original_text": AR_prediction_all[i],
                                    "allsat": best_allsat[i].item(),
                                    "losses": best_losses[i,:].tolist(),
                                    "weighted_loss": best_weighted_loss[i].item(),
                                    "edited": edited_at_all_yn[i].tolist(),
                                } for i in range(len(AR_prediction_all))
                            ],
                        }
            
                json.dump(output, outfs[_iter])
                outfs[_iter].write("\n")
                outfs[_iter].flush()
                iteration_specific_execution_times[_iter] += 0
        
        else:
            
            num_edited += edit_yn.sum().item()
            num_skipped += (len(AR_prediction_all) - edit_yn.sum().item())
            num_decoded_tokens += sum([len(x) for x in name2tokenizer[config["tokenizer_paths"][0]](running_text, add_special_tokens=False).input_ids])       
            
            for _iter in range(config['n_iter']):
                iter_start_time = time.time()
                if sum([1 if x != "" else 0 for x in running_text]) == 0:
                    # corner case: after deletion is introduced, sometimes all tokens are deleted and only "" remains. this occurs when initial sequence length is short.
                    print(f"ending iterations")
                    
                    for _iter_future in range(_iter, config['n_iter']):
                        # if corner case occurs & end iterations, 
                        # save result from previous iter for the current and the rest of iterations
                        output = {
                            "prompt": {
                                "text": source_text,
                            },
                            "generations": [
                                {
                                    "text": best_text[i],
                                    "original_text": AR_prediction_all[i],
                                    "allsat": best_allsat[i].item(),
                                    "losses": best_losses[i,:].tolist(),
                                    "weighted_loss": best_weighted_loss[i].item(),
                                    "edited": edited_at_all_yn[i].tolist(),
                                } for i in range(len(AR_prediction_all))
                            ],
                        }
            
                        json.dump(output, outfs[_iter_future])
                        outfs[_iter_future].write("\n")
                        outfs[_iter_future].flush()
                        iteration_specific_execution_times[_iter_future] += 0
                    
                    break
                
                ## masked_text : N (num samples to edit)
                if config["task"] == "nli":
                    sequences = [locator.tokenizer.bos_token + source_text + locator.tokenizer.sep_token + h + locator.tokenizer.eos_token for h in running_text]
                    tokenized_sequences = locator.tokenizer(sequences, add_special_tokens=False,padding=True, truncation=True, return_tensors='pt').to(config['device'])
       
                    masked_text = locator.locate_main(tokenized_sequences, 
                                            method = config['locate_method'], # grad_norm
                                            max_num_tokens = config['num_edit_token_per_step'], # 7
                                            unit = config['locate_unit'], # word
                                            num_layer = 10,#-2, #penultimate
                                            label_id = config['target_label_ids'][1],
                                            tokenized_input=True,
                                            use_energy=False)
                else:
                    masked_text = locator.locate_main(running_text, 
                                            method = config['locate_method'], # grad_norm
                                            max_num_tokens = config['num_edit_token_per_step'], # 7
                                            unit = config['locate_unit'], # word
                                            num_layer = 10,#-2, #penultimate
                                            label_id = config['target_label_ids'][1],
                                            use_energy=False)

                span_lengths_es = []
                for test_sent in masked_text:
                    _, span_lengths = analyze_span_lengths_and_count(test_sent)
                    span_lengths_es.append(span_lengths)

                # Arguments
                final_hypotheses_ = []
                new_best_weighted_loss_ = []
                new_best_allsat_ = []
                new_best_logging_loss_ = []
                
                tmp_masked_text = []
                tmp_running_text = []
                edit_ixes_before_marking = edit_yn.nonzero().squeeze(-1)
                for idx in range(len(masked_text)):
                    test_sent = masked_text[idx]
                    test_sent_span_lengths = span_lengths_es[idx]
                    if len(test_sent_span_lengths) == 0:
                        # corner case: when text is short, locator sometimes returns no mask. (this occurs when length is smaller than 3.)
                        edit_yn[edit_ixes_before_marking[idx]] = False
                        continue                    
                    final_hypotheses_curr, new_best_weighted_loss_curr, new_best_allsat_curr, new_best_logging_loss_curr = \
                        editing_with_delete_variable_replace(source_text, test_sent, test_sent_span_lengths, mlm, mlm_tokenizer, lossfns, config, batch_size=32)
                    final_hypotheses_.extend(final_hypotheses_curr)
                    new_best_weighted_loss_.append(new_best_weighted_loss_curr)
                    new_best_allsat_.append(new_best_allsat_curr)
                    new_best_logging_loss_.append(new_best_logging_loss_curr)
                    tmp_masked_text.append(test_sent)
                    tmp_running_text.append(running_text[idx])
                
                masked_text = tmp_masked_text
                running_text = tmp_running_text
                if len(new_best_weighted_loss_) == 0:
                    new_best_weighted_loss_ = torch.empty((0,)).to(config['device'])
                    new_best_allsat_ = torch.empty((0,)).bool().to(config['device'])
                    new_best_logging_loss_ = torch.empty((0, len(config['losses']))).to(config['device'])
                else:
                    new_best_weighted_loss_ = torch.cat(new_best_weighted_loss_)
                    new_best_allsat_ = torch.cat(new_best_allsat_)
                    new_best_logging_loss_ = torch.cat(new_best_logging_loss_, dim=0)
                

                new_best_weighted_loss = torch.empty((len(AR_prediction_all),)).fill_(float("inf")).to(config['device'])
                new_best_weighted_loss[edit_yn] = new_best_weighted_loss_
                
                new_best_logging_loss = torch.empty((len(AR_prediction_all), len(config['losses']))).fill_(float("inf")).to(config['device'])
                new_best_logging_loss[edit_yn, :] = new_best_logging_loss_
                
                new_best_allsat = torch.zeros((len(AR_prediction_all),)).bool().to(config['device'])
                new_best_allsat[edit_yn] = new_best_allsat_
                edit_ixes = edit_yn.nonzero().squeeze(-1)
                final_hypotheses = [final_hypotheses_[torch.where(edit_ixes==i)[0].item()] if edit_yn[i] else '' for i in range(len(AR_prediction_all))]
                
                update = torch.Tensor([]).bool().to(config['device'])
                if config['selection_criteria'] == "weighted_sum":
                    update = best_weighted_loss > new_best_weighted_loss ## edit_yn이 false 였던 곳은 무조건 false
                elif config['selection_criteria'] == "allsat_primary":
                    update = (~best_allsat & new_best_allsat) | \
                            (~best_allsat & ~new_best_allsat & (best_weighted_loss > new_best_weighted_loss)) | \
                            (best_allsat & new_best_allsat & (best_losses[:, 0] > new_best_logging_loss[:, 0])) 
                            ## (~best_allsat & new_best_allsat) : edit_yn이 false였던 곳은 무조건 false
                            ## (~best_allsat & ~new_best_allsat & (best_weighted_loss > new_best_weighted_loss)) : edit_yn이 false 였던 곳은 무조건 false
                            ## (best_allsat & new_best_allsat & (best_losses[:, 0] > new_best_logging_loss[:, 0])) : edit_yn이 false였던 곳은 무조건 false
                update = (update & edit_yn) # edit 대상인 것들만 update하기 위해서 update 조건에 edit_yn을 sum.

                ## intermediate output for debugging
                for sample_ix in range(len(running_text)): # edit 대상인 것들만 update.
                    int_output[edit_ixes[sample_ix]].update({f"iter{_iter}_original_sentence": running_text[sample_ix],
                                                            f"iter{_iter}_masked_sentence": masked_text[sample_ix],
                                                            f"iter{_iter}_best_text": final_hypotheses[edit_ixes[sample_ix]],
                                                            f"iter{_iter}_update": update[edit_ixes[sample_ix]].item()})    
                
                # update running_text, best_text, best_allsat, best_losses, best_weighted_loss
                for update_index in update.nonzero().squeeze(-1).tolist():
                    best_text[update_index] = final_hypotheses[update_index]
                best_allsat[update] = new_best_allsat[update]
                best_losses[update] = new_best_logging_loss[update]
                best_weighted_loss[update] = new_best_weighted_loss[update]

                output = {
                            "prompt": {
                                "text": source_text,
                            },
                            "generations": [
                                {
                                    "text": best_text[i],
                                    "original_text": AR_prediction_all[i],
                                    "allsat": best_allsat[i].item(),
                                    "losses": best_losses[i,:].tolist(),
                                    "weighted_loss": best_weighted_loss[i].item(),
                                    "edited": edited_at_all_yn[i].tolist(),
                                } for i in range(len(AR_prediction_all))
                            ],
                        }
            
                json.dump(output, outfs[_iter])
                outfs[_iter].write("\n")
                outfs[_iter].flush()
                iteration_specific_execution_times[_iter] += time.time() - iter_start_time

                es_patience_count[(best_allsat & edit_yn).nonzero().squeeze(-1)] += 1

                if (config["early_stopping_patience"] != -1):
                    edit_yn[es_patience_count > config['early_stopping_patience']] = False
                if edit_yn.sum() == 0:
                    for _iter_future in range(_iter+1, config['n_iter']):
                        # if early stop, save current iteration's result for the rest of iterations
                        output = {
                            "prompt": {
                                "text": source_text,
                            },
                            "generations": [
                                {
                                    "text": best_text[i],
                                    "original_text": AR_prediction_all[i],
                                    "allsat": best_allsat[i].item(),
                                    "losses": best_losses[i,:].tolist(),
                                    "weighted_loss": best_weighted_loss[i].item(),
                                    "edited": edited_at_all_yn[i].tolist(),
                                } for i in range(len(AR_prediction_all))
                            ],
                        }
            
                        json.dump(output, outfs[_iter_future])
                        outfs[_iter_future].write("\n")
                        outfs[_iter_future].flush()
                        iteration_specific_execution_times[_iter_future] += 0
                    break
                
            
                running_text = [x for i, x in enumerate(final_hypotheses) if edit_yn[i]]
        
                
        if (time.time() - main_start_time) > config['server_time_limit'] * 60 * 60 * 0.9:
            interrupted = True
            break

    for _iter in range(config['n_iter']):
        outfs[_iter].close()
    # outf.close()
    # int_outf.close()

    
    if config["resume"]:
        try: 
            run.summary["decode_time"]+= time.time() - decode_start_time
        except:
            run.summary["decode_time"]= time.time() - decode_start_time
    else:
        run.summary["decode_time"] = time.time() - decode_start_time
    run.summary['num_decoded_tokens'] = num_decoded_tokens
    run.summary['toks_p_sec'] = (num_decoded_tokens/run.summary['decode_time'])
    run.summary["num_skipped"] = num_skipped
    run.summary["num_edited"] = num_edited

    # save approx. decoding time for each # of iteration 
    # note that decoding time for 3 iterations include decoding time for 1st, 2nd, and 3rd iteration.
    # thus, suppose that max # of iterations is 10 and we want to find out decoding time for 3 iterations,
    # we subtract decoding time for 4th to 10th iteration from total decoding time.
    for _iter in range(config['n_iter']):
        iter_decode_time = run.summary["decode_time"]
        for _iter_future in range(_iter+1, config['n_iter']):
            iter_decode_time -= iteration_specific_execution_times[_iter_future]
        run.summary[f'iter{_iter}_decode_time'] = iter_decode_time

    run.finish()
    
    ## delete loss functions to clear up gpu memory
    try:
        del lossfns, name2tokenizer, name2model, name2config, loss2tokenizer
    except:
        pass
    torch.cuda.empty_cache()
    
    if (not interrupted):
        for _iter in range(config['n_iter']):
            if config["task"] == "toxicity":
                run_generation_evaluation(
                    "",
                    outfile+f".{_iter}",
                    "toxicity,toxicity-int,ppl-qwen,dist-n,repetition,fluency,contents-preservation,h1",
                    toxicity_model_path=config["model_paths"][1],
                    toxicity_model_type=config["model_types"][1],
                    source_file_path=config["source_data"],
                    task=config["task"],
                    target_style=config["target_style"]
                )  # 시간 문제로, perspective api 제외
            elif config["task"] == "formality":
                run_generation_evaluation(
                    "",
                    outfile+f".{_iter}",
                    "formality-int,formality-ext,ppl-qwen,dist-n,repetition,fluency,contents-preservation,h1", 
                    formality_model_path=config["model_paths"][1],
                    formality_model_type=config["model_types"][1],
                    source_file_path=config["source_data"],
                    task=config["task"],
                    target_style=config["target_style"]
                )
            elif config["task"] == "sentiment":
                run_generation_evaluation(
                    "",
                    outfile+f".{_iter}",
                    "sentiment-int,sentiment-ext,ppl-qwen,dist-n,repetition,fluency,contents-preservation,h1",
                    sentiment_model_path=config["model_paths"][1],
                    sentiment_model_type=config["model_types"][1],
                    source_file_path=config["source_data"],
                    task=config["task"],
                    target_style=config["target_style"]
                )
            elif config["task"] == "nli":
                run_generation_evaluation(
                    "",
                    outfile+f".{_iter}",
                    "nli,ppl-qwen,dist-n,repetition,fluency,contents-preservation,h1",
                    source_file_path=config["source_data"],
                    task=config["task"],
                    target_style=config["target_style"]
                )  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locally Editing Text Generation")
    parser.add_argument(
        "--task",
        type=str,
        help="task name",
        choices=["toxicity", "formality", "sentiment", "nli"],
    )
    parser.add_argument(
        "--source_data",
        type=str,
        help="source data path",
    )
    parser.add_argument(
        "--source_style", type=str, help="source style. e.g. toxic"
    )
    parser.add_argument(
        "--target_style", type=str, help="target style. e.g. nontoxic"
    )
    parser.add_argument(
        "--target_label_ids",
        nargs="+",
        type=int,
        help="a list of indices of target label used in each of models. e.g. [1,1]",
    )
    parser.add_argument(
        "--model_paths",
        nargs="+",
        type=str,
        help="model paths for energy functions. the first has to be a language model for measuring fluency. e.g. gpt2-large, <path-to-your-energy-function>",
    )
    parser.add_argument(
        "--tokenizer_paths",
        nargs="+",
        type=str,
        help="tokenizer paths. e.g. gpt2-large, <path-to-your-energy-function>",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        type=str,
        default=["AutoModelForCausalLM", "AutoModelForSequenceClassification"],
        help="model types",
    )
    parser.add_argument(
        "--output_dir_prefix",
        type=str,
        help="output directory prefix. e.g. outputs/toxicity/",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="early stopping patience",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mlm-beamsearch-v0",
        help="method name for reranking. currently only support mlm-beamsearch-v0",
        choices=[
            "mlm-beamsearch-v0",
            "mlm-beamsearch-v1",
            "mlm-beamsearch-v2",
            "mlm-reranking",
        ],
    )
    parser.add_argument(
        "--locate_unit", type=str, default="word", help="unit to locate"
    )
    parser.add_argument(
        "--min_epsilons", nargs="+", type=float, default=[0.75], help="a list of threshold values for constraint energy functions other than fluency. in probability scale."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="number of samples to edit per prompt. This becomes the batch size for decoding.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--cache_dir", type=str, default="~/hf_cache", help="cache directory"
    )
    parser.add_argument(
        "--jsonl_primary_key", type=str, default="prompt", help="jsonl primary key"
    )
    parser.add_argument(
        "--jsonl_secondary_key", type=str, default="text", help="jsonl secondary key"
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        type=str,
        default=["gpt2", "classification_no_prefix_logprobloss"],
        help="losses",
    )
    parser.add_argument("--loss_weights", nargs="+", type=float, default=[1,1], help="closs weight")
    
    parser.add_argument(
        "--num_edit_token_per_step",
        type=int,
        default=7,
        help="number of edit tokens per step",
    )
    parser.add_argument(
        "--max_tokens_per_span",
        type=int,
        default=3,
        help="max number of tokens to replace each mask span (overrided if original span length was longer than this)",
    )
    parser.add_argument(
        "--consider_prompt_for_cand_gen",
        type=bool,
        default=True,
        help="whether to consider source_text when generating token-level candidates",
    )
    
    parser.add_argument("--k_per_location", type=int, default=10, help="k per location")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations")
    parser.add_argument(
        "--selection_criteria",
        type=str,
        default="allsat_primary",
        help="selection criteria",
        choices=[
            "weighted_sum",
            "allsat_primary"
        ],
    )
    parser.add_argument("--beam_size", type=int, default=5, help="beam size")
    parser.add_argument(
        "--wandb_project", type=str, default="mlm_reranking", help="wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="hayleyson", help="wandb entity name"
    )
    parser.add_argument("--wandb_run_id", type=str, help="wandb run name")
    parser.add_argument(
        "--resume", action="store_true", help="whether to resume from a previous run"
    )
    parser.add_argument("--slurm_job_id", type=str, help="slurm job id (for debugging)")
    parser.add_argument(
        "--dont_skip_allsat",
        action="store_true",
        help="if this argument is passed, the module will conduct decoding on all samples even if they already satisfy constraints",
    )
    parser.add_argument(
        "--locate_method",
        type=str,
        help="method to use for locating tokens",
        choices=["attention", "grad_norm"],
        default="grad_norm",
    )
    parser.add_argument(
        "--server_time_limit",
        type=float,
        help="Number of maximum hours to run the script for. Can be fractions e.g. 7.5.",
        default=10000
    )

    args = parser.parse_args()
    config = vars(args)

    # Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
    # sweep_config = {
    #     'method': 'grid', #grid, random
    #     'metric': {
    #     'name': 'h1',
    #     'goal': 'maximize'   
    #     },
    #     'parameters': {
    #         'closs_weight': {
    #             'values':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #         },
    #     }
    # }
    
    # sweep_config = {
    #     'method': 'grid', #grid, random
    #     'metric': {
    #     'name': 'h1',
    #     'goal': 'maximize'   
    #     },
    #     'parameters': {
    #         'k_per_location': {
    #             'values':[5, 10, 15]
    #         },
    #         'beam_size': {
    #             'values':[3, 5, 7]
    #         },
    #     }
    # }
    
    # sweep_id = wandb.sweep(sweep_config, entity=config['wandb_entity'], project=config['wandb_project'])
    # sw_count = math.prod([len(val['values']) for val in sweep_config['parameters'].values()])
    # logger.info(f"Number of sweeps: {sw_count}")
    # main_for_sweep = functools.partial(main, config)
    # wandb.agent(sweep_id, function=main_for_sweep, count=sw_count)
    main(config)
