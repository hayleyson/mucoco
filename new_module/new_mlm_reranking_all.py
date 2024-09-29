#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from itertools import chain
import math
import argparse
import json
import logging
import os
import time
# os.chdir('/data/hyeryung/mucoco')
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
from new_module.new_decode_utils import get_beam_hypotheses_v0, get_beam_hypotheses_v1, get_combi_hypotheses, final_reranking
from new_module.evaluation.evaluate_wandb import evaluate_main
from new_module.locate.new_locate_utils import LocateMachine
from new_module.utils.robertacustom import RobertaCustomForSequenceClassification

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGGING_LEVEL", logging.DEBUG))


def main(config):
    
    main_start_time = time.time()

    if not config.get("model_tag", None):
        if "energy-training" in config["model_paths"][1]:
            config["model_tag"] = "em"
        else:
            config["model_tag"] = "clsf"

        if (config["task"] == "formality") and ("gyafc" in config["model_paths"][1]):
            config["model_tag"] += "-gyafc"

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

    run_id = run.path.split("/")[-1]
    display_name = f"{run_id}"
    

    outdir = os.path.join(config["output_dir_prefix"], display_name)
    os.makedirs(outdir, exist_ok=True)
    outfile = f"{outdir}/outputs_epsilon{config['min_epsilons'][0]}.txt"
    run.summary["outfile_path"] = outfile

    class dummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    build_loss_args = dummyArgs(**config["build_loss_dict"])

    ## load data
    if (config["task"] == "toxicity") or (config["task"] == "sentiment"):
        source_dataset = [
            json.loads(l)[config["jsonl_primary_key"]][config["jsonl_secondary_key"]]
            for l in open(config["source_data"])
        ]
        generation_dataset = [
            json.loads(l)["generations"] for l in open(config["source_data"])
        ]
    elif (config["task"] == "formality") or (config["task"] == "sentiment-lewis-compr"):
        with open(config["source_data"], "r") as f:
            generation_dataset = [line.rstrip('\n') for line in f.readlines()]
        source_dataset = ["" for l in generation_dataset]

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

    ## load tokenizer, models, define losses
    name2tokenizer = {}
    name2model = {}
    name2config = {}
    loss2tokenizer = {}
    embed_luts = []

    for i, model_path in enumerate(config["model_paths"]):
        if (
            model_path not in name2model
        ):  # making sure we are not loading the model twice in case some constraints use the same model.
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

        input_embeds = name2model[model_path].get_input_embeddings()
        if isinstance(input_embeds, torch.nn.Sequential):
            input_embeds = input_embeds[0]
        embed_luts.append(input_embeds)

        if config["target_type"] == "embeds":
            embed_luts[-1].requires_grad = False

    mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    mlm = None if config["method"] == "mlm-beamsearch-v2" else AutoModelForMaskedLM.from_pretrained("roberta-base").to(config['device'])

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
    # lossfns[0].tokenizer = loss2tokenizer[config["losses"][0]]
    # lossfns[1].tokenizer = loss2tokenizer[config["losses"][1]]

    # define an object to locate problematic phrases
    locator = LocateMachine(lossfns[1].model, lossfns[1].tokenizer)

    label_ids = config["target_label_ids"]  # target label's ids for each loss

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

    # loss_weights = [1 - wandb.config.closs_weight, wandb.config.closs_weight]
    loss_weights = config['loss_weights']
    interrupted = False
    if (config["task"] == "toxicity") or (config["task"] == "sentiment"):
        text_id_interval = 1
    elif (config["task"] == "formality") or (
            config["task"] == "sentiment-lewis-compr"
        ):
        text_id_interval = config['num_samples']
    for text_id in range(resume_idx, len(source_dataset), text_id_interval):
        source_text = source_dataset[text_id]
        if source_text == "":
            source_text = lossfns[0].tokenizer.bos_token

        if (config["task"] == "toxicity") or (config["task"] == "sentiment"):
            AR_prediction_all = [x["text"] for x in generation_dataset[text_id]]
            # predicted_batches = [x["tokens"] for x in generation_dataset[text_id]]
            # predicted_batches = [
            #     torch.tensor([x], dtype=torch.long, device=config["device"])
            #     for x in predicted_batches
            # ]
            
        elif (config["task"] == "formality") or (
            config["task"] == "sentiment-lewis-compr"
        ):
            # AR_prediction_all = [generation_dataset[text_id]]
            AR_prediction_all = generation_dataset[text_id: text_id + text_id_interval]
 
        curr_num_samples = len(AR_prediction_all)
        if curr_num_samples == 0:
            continue
        # for sample_idx in range(config["num_samples"])[:]:

        ######### change here! instead of for loop, do a batched operation ########

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
            curr_loss += loss_weights[lossid] * lossvalue
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
        running_text = [x for i, x in enumerate(AR_prediction_all) if edit_yn[i]] ## 실제 고쳐야 할 sample만 가지고 있음
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
            
            for _iter in range(wandb.config.n_iter):
                ## masked_text : N (num samples to edit)
                masked_text = locator.locate_main(running_text, 
                                        method = config['locate_method'], 
                                        max_num_tokens = wandb.config.num_edit_token_per_step, 
                                        unit = config['locate_unit'], 
                                        num_layer = 10,#-2, #penultimate
                                        label_id = config['target_label_ids'][1])

                ## replace tokens at the indices with mask tokens
                
                inputs = mlm_tokenizer(
                    masked_text, return_tensors="pt", padding=True, truncation=True
                )
                inputs = inputs.to(config['device']) 
                masked_sequence=inputs['input_ids']


                ## make predictions for the masked indices
                with torch.no_grad():
                    logits = mlm(**inputs).logits

                special_token_ids = mlm_tokenizer.convert_tokens_to_ids(mlm_tokenizer.all_special_tokens)
                logits[:, :, special_token_ids] = -float("inf")

                
                indices_in_mlm_tokens = (
                    inputs.input_ids == mlm_tokenizer.mask_token_id
                ).nonzero(as_tuple=True)

                ## get top k tokens for each index
                predicted_token_ids = torch.topk(
                    logits[indices_in_mlm_tokens[0], indices_in_mlm_tokens[1], :],
                    k=config['k_per_location'],
                    dim=-1,
                )

                
                if config["method"] == "mlm-beamsearch-v0":
                    hypotheses = get_beam_hypotheses_v0(source_text, 
                            masked_sequence, 
                            indices_in_mlm_tokens,
                            predicted_token_ids.indices,
                            mlm_tokenizer, 
                            lossfns,
                            config)
                elif config["method"] == "mlm-beamsearch-v1":
                    hypotheses = get_beam_hypotheses_v1(source_text, 
                            masked_sequence, 
                            indices_in_mlm_tokens,
                            predicted_token_ids.indices,
                            mlm_tokenizer, 
                            lossfns,
                            config)
                elif config["method"] == "mlm-reranking":
                    hypotheses = get_combi_hypotheses(masked_sequence, 
                                indices_in_mlm_tokens,
                                predicted_token_ids.indices,
                                mlm_tokenizer,
                                config)

                    
                    
                final_hypotheses_, new_best_weighted_loss_, new_best_allsat_, new_best_logging_loss_ = final_reranking(source_text,
                                                                                                                    hypotheses,
                                                                                                                    lossfns,
                                                                                                                    config,
                                                                                                                    batch_size=64)


                ## final_hypotheses, new_best_weighted_loss, new_best_allsat, new_best_logging_loss 모두 N 의 길이를 가짐 
                ## 특히 edit 대상이 iteration마다 달라지면 best_... tensor와 new_best_... tensor간에 크기가 달라서 아래 코드 실행시 에러가 날 것이다.
                
                new_best_weighted_loss = torch.empty((len(AR_prediction_all),)).fill_(float("inf")).to(config['device'])
                new_best_weighted_loss[edit_yn] = new_best_weighted_loss_
                
                new_best_logging_loss = torch.empty((len(AR_prediction_all), len(config['losses']))).fill_(float("inf")).to(config['device'])
                new_best_logging_loss[edit_yn, :] = new_best_logging_loss_
                
                new_best_allsat = torch.zeros((len(AR_prediction_all),)).bool().to(config['device'])
                new_best_allsat[edit_yn] = new_best_allsat_
                edit_ixes = edit_yn.nonzero().squeeze(-1)
                final_hypotheses = [final_hypotheses_[torch.where(edit_ixes==i)[0].item()] if edit_yn[i] else '' for i in range(len(AR_prediction_all))]
                
                update = torch.Tensor([]).bool().to(config['device'])
                if wandb.config.selection_criteria == "weighted_sum":
                    update = best_weighted_loss > new_best_weighted_loss ## edit_yn이 false 였던 곳은 무조건 false
                elif wandb.config.selection_criteria == "allsat_primary":
                    update = (~best_allsat & new_best_allsat) | \
                            (~best_allsat & ~new_best_allsat & (best_weighted_loss > new_best_weighted_loss)) | \
                            (best_allsat & new_best_allsat & (best_losses[:, 0] > new_best_logging_loss[:, 0])) 
                            ## (~best_allsat & new_best_allsat) : edit_yn이 false였던 곳은 무조건 false
                            ## (~best_allsat & ~new_best_allsat & (best_weighted_loss > new_best_weighted_loss)) : edit_yn이 false 였던 곳은 무조건 false
                            ## (best_allsat & new_best_allsat & (best_losses[:, 0] > new_best_logging_loss[:, 0])) : edit_yn이 false였던 곳은 무조건 false
                update = (update & edit_yn) # edit 대상인 것들만 update하기 위해서 update 조건에 edit_yn을 sum.

                ## intermediate output for debugging
                # for sample_ix in edit_yn.nonzero().squeeze(-1).tolist(): # edit 대상인 것들만 update.
                
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
        run.summary["decode_time"] += time.time() - decode_start_time
    else:
        run.summary["decode_time"] = time.time() - decode_start_time
    run.summary['num_decoded_tokens'] = num_decoded_tokens
    run.summary['toks_p_sec'] = (num_decoded_tokens/run.summary['decode_time'])
    run.summary["num_skipped"] = num_skipped
    run.summary["num_edited"] = num_edited

    run.finish()
    
    ## delete loss functions to clear up gpu memory
    try:
        del lossfns, name2tokenizer, name2model, name2config, loss2tokenizer
    except:
        pass
    torch.cuda.empty_cache()
    
    if (not interrupted):
        if config["task"] == "toxicity":
            evaluate_main(
                run.path,
                outfile,
                "toxicity,toxicity-int,ppl-big,dist-n,repetition,fluency,contents-preservation,qual",
                toxicity_model_path=config["model_paths"][1],
                toxicity_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )  # 시간 문제로, perspective api 제외
        elif config["task"] == "formality":
            evaluate_main(
                run.path,
                outfile,
                "formality-int,formality-ext,ppl-big,dist-n,repetition,fluency,contents-preservation,qual",
                formality_model_path=config["model_paths"][1],
                formality_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )
        elif config["task"] == "sentiment":
            evaluate_main(
                run.path,
                outfile,
                "sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency,contents-preservation,qual",
                sentiment_model_path=config["model_paths"][1],
                sentiment_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )
        elif config["task"] == "sentiment-lewis-compr":
            evaluate_main(
                run.path,
                outfile,
                "sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency,contents-preservation,qual",
                sentiment_model_path=config["model_paths"][1],
                sentiment_model_type=config["model_types"][1],
                source_file_path=config["source_data"]
            )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locally Editing Text Generation")
    parser.add_argument(
        "--task",
        type=str,
        help="task name",
        choices=["toxicity", "formality", "sentiment", "sentiment-lewis-compr"],
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
        "--min_epsilons", nargs="+", type=float, default=[0.75], help="min epsilons"
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
        "--cache_dir", type=str, default="hf_cache", help="cache directory"
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
        "--build_loss_dict",
        type=json.loads,
        default='{"coeff_steps": 200, "coeff_pattern": "constant", "loss_type": "xentropy", "length_normalize": false, "AR_temperature": 1.0, "AR_top_k": 0, "AR_top_p": 0.96, "max_output_length": 20}',
        help="build loss dict",
    )
    parser.add_argument(
        "--num_edit_token_per_step",
        type=int,
        default=5,
        help="number of edit tokens per step",
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

    args = parser.parse_args()
    config = vars(args)

    main(config)
