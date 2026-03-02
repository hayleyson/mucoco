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

from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
import wandb

from ebl_edit.edit.mlm.mlm_edit import analyze_span_lengths_and_count, editing_with_delete_variable_replace
from ebl_edit.evaluation.evaluate_wandb import evaluate_main
from ebl_edit.locate.locate import EnergyBasedLocator
from ebl_edit.energy_models import CausalLMEnergyModel, DiscriminatorEnergyModel, calculate_energies
from ebl_edit.energy_training.nli.models import EncoderModel
from ebl_edit.base_lms import HFVanillaLM, HFChatLM, OpenAIChatLM

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGGING_LEVEL", logging.DEBUG))


load_dotenv(find_dotenv(), override=True)


def main(config):
    
    main_start_time = time.time()

    if not config.get("model_tag", None):
        if ("energy-training" in config["model_paths"][1]) or ("finegrained_labels" in config["model_paths"][1]): 
            config["model_tag"] = "em"
        else:
            config["model_tag"] = "clsf"

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
    outfile = f"{outdir}/outputs_epsilon{config['epsilon'][0]}.txt"
    run.summary["outfile_path"] = outfile


    ## load data
    if (config["task"] == "toxicity") or (config["task"] == "sentiment") or (config["task"] == "nli"):
        source_dataset = [
            json.loads(l)[config["jsonl_primary_key"]][config["jsonl_secondary_key"]]
            for l in open(config["source_data"])
        ]
        if config["load_pregen_outputs"]:
            generation_dataset = [
                json.loads(l)["generations"] for l in open(config["source_data"])
            ]
        else:
            generation_dataset = ["" for _ in range(len(source_dataset))]
    elif (config["task"] == "formality"):
        with open(config["source_data"], "r") as f:
            generation_dataset = [line.rstrip('\n') for line in f.readlines()]
        source_dataset = ["" for _ in range(len(generation_dataset))]

    ## create base LM instance if needed
    if not config["load_pregen_outputs"]:
        if "gpt" in config["base_lm"].lower() and "gpt2" not in config["base_lm"].lower(): 
            base_lm = OpenAILM(config["base_lm"])
        elif ("instruction" in config["base_lm"].lower()) or ("-it" in config["base_lm"].lower()) or ("chat" in config["base_lm"].lower()):
            base_lm = HFChatLM(config["base_lm"])
        else:
            base_lm = HFVanillaLM(config["base_lm"])
    else:
        base_lm = None

    # check if outfile exists
    if (config["resume"]) and (os.path.exists(outfile)):

        with open(outfile, "r") as f:
            existing_gens = [x.rstrip("\n") for x in f.readlines()]
        resume_idx = len(existing_gens)
        if resume_idx == len(source_dataset):
            logger.debug("output file is already complete. skipping this run.")
            return
        elif resume_idx < len(source_dataset):
            logger.info(
                f"output file already exists but is incomplete. resuming from index: {resume_idx}"
            )
            outf = open(outfile, "a")
            int_outf = open(outfile+".intermediate", "a")
        else:
            logger.critical(
                f"output file seems to be corrupted. The file length is {resume_idx}, where the size of source_dataset is {len(source_dataset)}"
            )
            return
    else:
        resume_idx = 0
        outf = open(outfile, "w")
        int_outf = open(outfile+".intermediate", "w")

    mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    mlm = None if config["method"] == "mlm-beamsearch-v2" else AutoModelForMaskedLM.from_pretrained("roberta-base").to(config['device'])

    energyfns = [None, None]
    # TODO: support more than 2 energy functions
    energyfns[0] = CausalLMEnergyModel(config["model_paths"][0], 
                                        args = {
                                            "length_normalize": config.get("length_normalize", True),
                                            "length_normalize_power": config.get("length_normalize_power", 1.0),
                                            "apply_chat_template": config.get("apply_chat_template", False),
                                        })
    energyfns[1] = DiscriminatorEnergyModel(config["model_paths"][1],
                                        args = {
                                            "label_id": config.get("label_id", None),
                                            "task": config.get("task", None),
                                        })


    # define an object to locate problematic phrases
    locator = EnergyBasedLocator(energyfns[1].model, energyfns[1].tokenizer, config['task'])

    if getattr(wandb.config, "closs_weight", None) is not None: ## closs_weight is used if sweep is used
        config["loss_weights"] = [1, wandb.config.closs_weight]
        run.config.update({"closs_weight": config["loss_weights"]}, allow_val_change=True)
    logger.info(f"loss_weights: {config['loss_weights']}")


    run.summary["prep_time"] = time.time() - main_start_time
    ## beginning of main logic
    decode_start_time = time.time()
    # text_id = 0
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
        
        
    for text_id in range(resume_idx, len(source_dataset), text_id_interval):
        source_text = source_dataset[text_id]
        if (source_text == "") and (energyfns[0].tokenizer.bos_token is not None):
            source_text = energyfns[0].tokenizer.bos_token
        elif (source_text == "") and (energyfns[0].tokenizer.bos_token is None):
            source_text = " "

        if (config["task"] == "toxicity") or (config["task"] == "sentiment") or (config["task"] == "nli"):
            
            if config["load_pregen_outputs"]:
                AR_prediction_all = [x["text"] for x in generation_dataset[text_id]]
            else:
                AR_prediction_all = base_lm.generate(source_text,
                                                    n = config["num_samples"],
                                                    max_new_tokens = config["max_new_tokens"],
                                                    top_p = config["top_p"],
                                                    )
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
        loss_weighted_sum, losses = calculate_energies(energyfns, 
                                                    source_text, 
                                                    AR_prediction_all, 
                                                    config)
        
        edit_yn = torch.ones(len(AR_prediction_all), dtype=torch.bool).to(config['device'])
        allsat = losses[:,1] < -math.log(config["epsilon"])
        allsat_ix = allsat.nonzero().squeeze(0)
        if (not config["dont_skip_allsat"]):
            edit_yn[allsat_ix] = False
        edited_at_all_yn = edit_yn.detach().clone()
        
        es_patience_count = torch.zeros(len(AR_prediction_all),dtype=torch.long).to(config['device'])
        best_allsat = allsat.detach().clone()
        best_losses = losses.detach().clone()
        best_weighted_loss = loss_weighted_sum.detach().clone()            
        best_text = deepcopy(AR_prediction_all)
        running_text = [x for i, x in enumerate(AR_prediction_all) if edit_yn[i]] ## hold only samples that need be edited
        int_output = [{} for _ in range(len(AR_prediction_all))]

        if (edit_yn.sum().item() == 0) and (not config["dont_skip_allsat"]):
            ## save data
            num_edited += 0
            num_skipped += len(AR_prediction_all)
            num_decoded_tokens += 0
            
            
            logger.info(
                    f"skipping this sample since it already satisfies constraint. {best_losses}"
                )
        
        else:
            
            num_edited += edit_yn.sum().item()
            num_skipped += (len(AR_prediction_all) - edit_yn.sum().item())
            num_decoded_tokens += sum([len(x) for x in name2tokenizer[config["tokenizer_paths"][0]](running_text, add_special_tokens=False).input_ids])       
            
            for _iter in range(config['n_iter']):
                if sum([1 if x != "" else 0 for x in running_text]) == 0:
                    # corner case: after deletion is introduced, sometimes all tokens are deleted and only "" remains. this occurs when initial sequence length is short.
                    print(f"ending iterations")
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
                new_best_losses_ = []
                
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
                    final_hypotheses_curr, new_best_weighted_loss_curr, new_best_allsat_curr, new_best_losses_curr = \
                        editing_with_delete_variable_replace(source_text, test_sent, test_sent_span_lengths, mlm, mlm_tokenizer, energyfns, config, batch_size=32)
                    final_hypotheses_.extend(final_hypotheses_curr)
                    new_best_weighted_loss_.append(new_best_weighted_loss_curr)
                    new_best_allsat_.append(new_best_allsat_curr)
                    new_best_losses_.append(new_best_losses_curr)
                    tmp_masked_text.append(test_sent)
                    tmp_running_text.append(running_text[idx])
                
                masked_text = tmp_masked_text
                running_text = tmp_running_text
                if len(new_best_weighted_loss_) == 0:
                    new_best_weighted_loss_ = torch.empty((0,)).to(config['device'])
                    new_best_allsat_ = torch.empty((0,)).bool().to(config['device'])
                    new_best_losses_ = torch.empty((0, len(config['losses']))).to(config['device'])
                else:
                    new_best_weighted_loss_ = torch.cat(new_best_weighted_loss_)
                    new_best_allsat_ = torch.cat(new_best_allsat_)
                    new_best_losses_ = torch.cat(new_best_losses_, dim=0)
                

                ## final_hypotheses, new_best_weighted_loss, new_best_allsat, new_best_losses 모두 N 의 길이를 가짐 
                ## 특히 edit 대상이 iteration마다 달라지면 best_... tensor와 new_best_... tensor간에 크기가 달라서 아래 코드 실행시 에러가 날 것이다.
                
                new_best_weighted_loss = torch.empty((len(AR_prediction_all),)).fill_(float("inf")).to(config['device'])
                new_best_weighted_loss[edit_yn] = new_best_weighted_loss_
                
                new_best_losses = torch.empty((len(AR_prediction_all), len(config['losses']))).fill_(float("inf")).to(config['device'])
                new_best_losses[edit_yn, :] = new_best_losses_
                
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
                            (best_allsat & new_best_allsat & (best_losses[:, 0] > new_best_losses[:, 0])) 
                            ## (~best_allsat & new_best_allsat) : edit_yn이 false였던 곳은 무조건 false
                            ## (~best_allsat & ~new_best_allsat & (best_weighted_loss > new_best_weighted_loss)) : edit_yn이 false 였던 곳은 무조건 false
                            ## (best_allsat & new_best_allsat & (best_losses[:, 0] > new_best_losses[:, 0])) : edit_yn이 false였던 곳은 무조건 false
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
                best_losses[update] = new_best_losses[update]
                best_weighted_loss[update] = new_best_weighted_loss[update]

                es_patience_count[(best_allsat & edit_yn).nonzero().squeeze(-1)] += 1

                if (config["early_stopping_patience"] != -1):
                    edit_yn[es_patience_count > config['early_stopping_patience']] = False
                if edit_yn.sum() == 0:
                    break
                
            
                running_text = [x for i, x in enumerate(final_hypotheses) if edit_yn[i]]
        

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
            
        intermediate_output = {
                "prompt": {
                    "text": source_text,
                },
                "generations": 
                    int_output
                ,
            }

        json.dump(output, outf)
        outf.write("\n")
        outf.flush()
        
        json.dump(intermediate_output, int_outf)
        int_outf.write("\n")
        int_outf.flush()
                
        if (time.time() - main_start_time) > config['server_time_limit'] * 60 * 60 * 0.9:
            interrupted = True
            break

    outf.close()
    int_outf.close()

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

    run.finish()
    
    ## delete loss functions to clear up gpu memory
    try:
        del energyfns, name2tokenizer, name2model, name2config, loss2tokenizer
    except:
        pass
    torch.cuda.empty_cache()
    
    if (not interrupted):
        if config["task"] == "toxicity":
            evaluate_main(
                run.path,
                outfile,
                "toxicity,toxicity-int,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
                toxicity_model_path=config["model_paths"][1],
                toxicity_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )  # 시간 문제로, perspective api 제외
        elif config["task"] == "formality":
            evaluate_main(
                run.path,
                outfile,
                "formality-int,formality-ext,ppl-qwen,dist-n,repetition,fluency,contents-preservation", 
                formality_model_path=config["model_paths"][1],
                formality_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )
        elif config["task"] == "sentiment":
            evaluate_main(
                run.path,
                outfile,
                "sentiment-int,sentiment-ext,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
                sentiment_model_path=config["model_paths"][1],
                sentiment_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )
        elif config["task"] == "sentiment-lewis-compr":
            evaluate_main(
                run.path,
                outfile,
                "sentiment-int,sentiment-ext,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
                sentiment_model_path=config["model_paths"][1],
                sentiment_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )
        elif config["task"] == "nli":
            evaluate_main(
                run.path,
                outfile,
                "nli,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
                source_file_path=config["source_data"]
            )  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locally Editing Text Generation")
    parser.add_argument(
        "--task",
        type=str,
        help="task name",
        choices=["toxicity", "formality", "sentiment", "sentiment-lewis-compr", "nli"],
    )
    parser.add_argument(
        "--source_data",
        type=str,
        default="data/formality/GYAFC_Corpus/Entertainment_Music/test/informal",
        help="source data path",
    )
    parser.add_argument(
        "--source_style", type=str, default="informal", help="source style"
    )
    parser.add_argument(
        "--target_style", type=str, default="formal", help="target style"
    )
    parser.add_argument(
        "--target_label_ids",
        nargs="+",
        type=int,
        default=[1, 1],
        help="a list of indices of target label used in each of models. e.g. [1,1]",
    )
    parser.add_argument(
        "--model_paths",
        nargs="+",
        type=str,
        default=[
            "gpt2-large",
            "/home/s3/hyeryung/data/loc_edit/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17",
        ],
        help="model paths",
    )
    parser.add_argument(
        "--tokenizer_paths",
        nargs="+",
        type=str,
        default=[
            "gpt2-large",
            "/home/s3/hyeryung/data/loc_edit/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-rescale/epoch_17",
        ],
        help="tokenizer paths",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        type=str,
        default=["AutoModelForCausalLM", "RobertaCustomForSequenceClassification"],
        help="model types",
    )
    parser.add_argument(
        "--output_dir_prefix",
        type=str,
        help="output directory prefix. e.g. outputs/formality/mlm-reranking",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        help="early stopping patience",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mlm-beamsearch-v0",
        help="method name",
        choices=[
            "mlm-beamsearch-v0",
            "mlm-beamsearch-v1",
            "mlm-beamsearch-v2",
            "mlm-reranking",
        ],
    )
    parser.add_argument(
        "--locate_unit", type=str, default="token", help="unit to locate"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.75, help="Threshold for the target constraint probability to exceed"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="number of samples to edit per prompt. This becomes the batch size for decoding.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--target_type",
        type=str,
        default="embeds",
        help="target type (embeds, simplex, probability) from prior work's code",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="/data/hyeryung/hf_cache", help="cache directory"
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
    parser.add_argument("--loss_weights", nargs="+", type=float, default=[0.1,1.0], help="closs weight")
    
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
    
    parser.add_argument("--k_per_location", type=int, default=15, help="k per location")
    parser.add_argument("--n_iter", type=int, default=3, help="number of iterations")
    parser.add_argument(
        "--selection_criteria",
        type=str,
        default="weighted_sum",
        help="selection criteria",
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
        default="attention",
    )
    parser.add_argument(
        "--server_time_limit",
        type=float,
        help="Number of maximum hours to run the script for. Can be fractions e.g. 7.5.",
        default=10000
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.96,
        help="top p value for sampling",
    )
    parser.add_argument(
        "--load_pregen_outputs",
        action="store_true",
        help="whether to load pregenerated base lm outputs",
    )
    parser.add_argument(
        "--base_lm",
        type=str,
        default="gpt2-large",
        help="base lm name. Either a huggingface model name or an openai model name",
    )

    args = parser.parse_args()
    config = vars(args)

   
    main(config)
