#!/usr/bin/env python
# coding: utf-8


import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

import new_module.losses as lossbuilder
import wandb
from new_module.decode_utils import (
    beam_rerank_v0,
    beam_rerank_v1,
    beam_rerank_v2,
    combi_rerank,
)
from new_module.evaluate_wandb import evaluate
from new_module.locate.locate_utils import locate_main
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
    display_name = f"{config['method']}-{config['locate_unit']}-nps{wandb.config.num_edit_token_per_step}-k{wandb.config.k_per_location}-beam{wandb.config.beam_size}-{wandb.config.selection_criteria}"
    display_name += f"-{config['source_style']}-to-{config['target_style']}"
    display_name += f"-{config['locate_method']}"
    display_name += f"-{run_id}"
    

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
                name2tokenizer[model_path] = AutoTokenizer.from_pretrained(
                    config["tokenizer_paths"][i],
                    cache_dir=config["cache_dir"],
                    use_fast=True,
                )
            except:
                name2tokenizer[model_path] = AutoTokenizer.from_pretrained(
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
            name2model[model_path].cuda()

        input_embeds = name2model[model_path].get_input_embeddings()
        if isinstance(input_embeds, torch.nn.Sequential):
            input_embeds = input_embeds[0]
        embed_luts.append(input_embeds)

        if config["target_type"] == "embeds":
            embed_luts[-1].requires_grad = False

    mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    mlm = None if config["method"] == "mlm-beamsearch-v2" else AutoModelForMaskedLM.from_pretrained("roberta-base")  

    lossfns = []
    for i, loss in enumerate(config["losses"]):
        lossfns.append(
            lossbuilder.build_loss(
                loss,
                name2model[config["model_paths"][i]],
                name2tokenizer[config["model_paths"][i]],
                build_loss_args,
            )
        )
        lossfns[i].tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})
        loss2tokenizer[loss] = lossfns[i].tokenizer
    # lossfns[0].tokenizer = loss2tokenizer[config["losses"][0]]
    # lossfns[1].tokenizer = loss2tokenizer[config["losses"][1]]

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

    interrupted = False
    for text_id in range(len(source_dataset))[resume_idx:]:
        source_text = source_dataset[text_id]
        if source_text == "":
            source_text = lossfns[0].tokenizer.bos_token

        if (config["task"] == "toxicity") or (config["task"] == "sentiment"):
            # AR_prediction_all = [x["text"] for x in generation_dataset[text_id]]
            predicted_batches = [x["tokens"] for x in generation_dataset[text_id]]
            predicted_batches = [
                torch.tensor([x], dtype=torch.long, device=config["device"])
                for x in predicted_batches
            ]
            
        elif (config["task"] == "formality") or (
            config["task"] == "sentiment-lewis-compr"
        ):
            AR_prediction_all = [generation_dataset[text_id]]

        sample_idx = 0
        for sample_idx in range(config["num_samples"])[:]:
            
            if (config["task"] == "toxicity") or (config["task"] == "sentiment"):
                predicted_batch = predicted_batches[sample_idx].cuda()
                AR_prediction = lossfns[0].tokenizer.batch_decode(predicted_batch)[0]
            else:
                AR_prediction = AR_prediction_all[sample_idx]

            logger.debug(
                f"text_id {text_id} sample_id {sample_idx} \n[prompt] {source_text} [text] {AR_prediction}"
            )

            # --------------------------------------------------------------------------------------------- #
            ## check whether initial text satisfies constraint
            allsat = True
            gold_losses = []
            for lossid, lossname in enumerate(config["losses"]):
                with torch.no_grad():
                    lossvalue = lossfns[lossid].compute_gold_loss(
                        source_text, AR_prediction,
                        label_id=label_ids[lossid],
                    )
                    
                gold_losses.append(lossvalue.squeeze().item())
                if (lossid >= 1) and (gold_losses[lossid] > -np.log(
                    config["min_epsilons"][lossid - 1]
                )):
                    allsat = False

            if (allsat) and (not config["dont_skip_allsat"]):
                logger.info(
                    f"skipping this sample since it already satisfies constraint. {gold_losses}"
                )
                num_skipped += 1
                if sample_idx == 0:
                    output = {
                        "prompt": {
                            "text": source_text,
                        },
                        "generations": [
                            {
                                "text": AR_prediction,
                                "indices": [[]],
                                "allsat": -1,
                                "losses": gold_losses,
                                "weighted_loss": -1,
                                "edited": False,
                            }
                        ],
                    }
                    intermediate_output = {
                        "prompt": {
                            "text": source_text,
                        },
                        "generations": [
                            {}
                        ],
                    }
                else:
                    output["generations"].append(
                        {
                            "text": AR_prediction,
                            "indices": [[]],
                            "allsat": -1,
                            "losses": gold_losses,
                            "weighted_loss": -1,
                            "edited": False,
                        }       
                    )
                    intermediate_output['generations'].append({})

                if sample_idx + 1 == config["num_samples"]:
                    json.dump(output, outf)
                    outf.write("\n")
                    outf.flush()

                    json.dump(intermediate_output, int_outf)
                    int_outf.write("\n")
                    int_outf.flush()

            else:
                num_edited += 1
                num_decoded_tokens += name2tokenizer[config["tokenizer_paths"][0]].encode(AR_prediction, return_tensors="pt", add_special_tokens=False).size(-1)
                es_patience_count = 0
                (
                    best_ix,
                    best_allsat,
                    best_losses,
                    best_weighted_loss,
                ) = None, None, None, None
                
                best_text = AR_prediction

                _iter = 0
                for _iter in range(wandb.config.n_iter):
                    ## locate tokens to edit
                    masked_text  = locate_main(best_text, 
                                            config["locate_method"], 
                                            name2model[config["model_paths"][1]], 
                                            name2tokenizer[config["tokenizer_paths"][1]], 
                                            max_num_tokens = 6, 
                                            unit=config["locate_unit"], 
                                            device="cuda", 
                                            label_id=config["target_label_ids"][1],
                                            num_layer=10)
                    logger.debug(f"iter {_iter}, sample_idx: {sample_idx}")
                    logger.debug(f"locate result: {masked_text}")
                    
                    if config["method"] == "mlm-beamsearch-v2":
                        pass
                    else:
                        ## replace tokens at the indices with mask tokens
                        inputs = mlm_tokenizer(
                            masked_text, return_tensors="pt"
                        )
                        # inputs = mlm_tokenizer(
                        #     source_text + ' ' + masked_text[0], return_tensors="pt", add_special_tokens=False
                        # )
                        
                        ## make predictions for the masked indices
                        with torch.no_grad():
                            logits = mlm(**inputs).logits
                        indices_in_mlm_tokens = (
                            inputs.input_ids == mlm_tokenizer.mask_token_id
                        )[0].nonzero(as_tuple=True)[0]
                        # print(f"indices_in_mlm_tokens: {indices_in_mlm_tokens}")
                        ## get top k tokens for each index
                        predicted_token_ids = torch.topk(
                            logits[0, indices_in_mlm_tokens],
                            k=wandb.config.k_per_location,
                            dim=-1,
                        )
                        # print(f"predicted_token_ids: {predicted_token_ids}")
                        # print(f"mlm_tokenizer.batch_decode(predicted_token_ids.indices): {mlm_tokenizer.batch_decode(predicted_token_ids.indices)}")
                        
                    if config["method"] == "mlm-beamsearch-v0":
                        # print(config["method"])
                        hypotheses = beam_rerank_v0(source_text,
                                                    inputs.input_ids,
                                                    indices_in_mlm_tokens,
                                                    predicted_token_ids,
                                                    mlm_tokenizer, 
                                                    lossfns,
                                                    config, 
                                                    beam_size = wandb.config.beam_size)
                    elif config["method"] == "mlm-beamsearch-v1":
                        hypotheses = beam_rerank_v1(source_text,
                                                    inputs.input_ids,
                                                    indices_in_mlm_tokens,
                                                    predicted_token_ids,
                                                    mlm_tokenizer, 
                                                    lossfns,
                                                    config, 
                                                    beam_size = wandb.config.beam_size)
                    elif config["method"] == "mlm-beamsearch-v2":
                        source_batch = lossfns[0].tokenizer(source_text, add_special_tokens=False, return_tensors="pt").input_ids.to(config['device'])
                        masked_sequence = lossfns[0].tokenizer(masked_text, add_special_tokens=False, return_tensors="pt").input_ids.to(config['device'])
                        hypotheses = beam_rerank_v2(
                            source_batch,
                            masked_sequence,
                            lossfns[0].model,
                            lossfns[0].tokenizer,
                            config,
                            beam_size=wandb.config.beam_size,
                        )
                    elif config["method"] == "mlm-reranking":
                        hypotheses = combi_rerank(inputs.input_ids, ## in mlm tokenizer's tokens
                            indices_in_mlm_tokens,
                            predicted_token_ids,
                            mlm_tokenizer,
                            config)

                    candidate_total_losses = []
                    candidate_primary_losses = []
                    candidate_losses_for_loggings = []
                    candidate_allsats = []
                    loss_weights = [1 - wandb.config.closs_weight, wandb.config.closs_weight]
                    for hyp in hypotheses:
                        curr_loss = 0.0
                        logging_loss = []
                        allsat = True
                        for lossid, lossname in enumerate(config["losses"]):
                            with torch.no_grad():
                                lossvalue = lossfns[lossid].compute_gold_loss(
                                    source_text, hyp,
                                    label_id=config['target_label_ids'][lossid],
                                )
                            curr_loss += loss_weights[lossid] * lossvalue.item()
                            logging_loss.append(lossvalue.item())
                            if lossid==0:
                                candidate_primary_losses.append(lossvalue.item())
                            elif (lossid >= 1) and (
                                lossvalue.item()
                                > -np.log(config["min_epsilons"][lossid - 1])
                            ):
                                allsat = False
                        candidate_total_losses.append(curr_loss)
                        candidate_losses_for_loggings.append(logging_loss)
                        candidate_allsats.append(allsat)


                    if wandb.config.selection_criteria == "weighted_sum":
                        best_ix = np.argmin(np.array(candidate_total_losses))
                    elif wandb.config.selection_criteria == "allsat_primary":
                        allsat_ix = np.where(np.array(candidate_allsats) == True)[0]
                        if len(allsat_ix) > 0:
                            best_ix = np.argmin(
                                np.array(candidate_primary_losses)[allsat_ix]
                            )  # select min primary loss among allsats
                            best_ix = allsat_ix[best_ix]
                        else:  # if no candidate satisfying constraints, default to weighted_sum
                            best_ix = np.argmin(np.array(candidate_total_losses))

                    if _iter == 0:  
                        ## intermediate output for debugging
                        int_output = {f"iter{_iter}_original_sentence": best_text,
                                      f"iter{_iter}_masked_sentence": masked_text,
                                      f"iter{_iter}_best_text": hypotheses[best_ix],
                                      f"iter{_iter}_update": True}                      
                        
                        ## save the best prediction in a format compatible with mucola outputs
                        best_text = hypotheses[best_ix]
                        best_allsat = candidate_allsats[best_ix]
                        best_losses = candidate_losses_for_loggings[best_ix]
                        best_weighted_loss = candidate_total_losses[best_ix]

                        logger.debug(f"best_text: {best_text}")
                        
                    else:
                        update = False
                        if wandb.config.selection_criteria == "weighted_sum":
                            if best_weighted_loss > candidate_total_losses[best_ix]:
                                update = True
                        elif wandb.config.selection_criteria == "allsat_primary":
                            if (
                                best_allsat is False
                                and candidate_allsats[best_ix] is True
                            ):
                                update = True
                            elif (
                                best_allsat is False
                                and candidate_allsats[best_ix] is False
                            ):
                                if best_weighted_loss > candidate_total_losses[best_ix]:
                                    update = True
                            elif (
                                best_allsat is True
                                and candidate_allsats[best_ix] is True
                            ):
                                if (
                                    best_losses[0]
                                    > candidate_losses_for_loggings[best_ix][0]
                                ):
                                    update = True


                        ## intermediate output for debugging
                        int_output |= {f"iter{_iter}_original_sentence": best_text,
                                      f"iter{_iter}_masked_sentence": masked_text,
                                      f"iter{_iter}_best_text": hypotheses[best_ix],
                                      f"iter{_iter}_update": update}    
    
                        if update:
                            ## save the best prediction in a format compatible with mucola outputs
                            best_text = hypotheses[best_ix]
                            best_allsat = candidate_allsats[best_ix]
                            best_losses = candidate_losses_for_loggings[best_ix]
                            best_weighted_loss = candidate_total_losses[best_ix]

                            logger.debug(f"iter {_iter}. Update best prediction")
                            logger.debug(f"best_text: {best_text}")
                        
                        if best_allsat:
                            es_patience_count += 1
                            if (config["early_stopping_patience"] != -1) and (es_patience_count > config["early_stopping_patience"]):
                                logger.info(f"early stopping at iter {_iter}")
                                break

                if sample_idx == 0:
                    output = {
                        "prompt": {
                            "text": source_text,
                        },
                        "generations": [
                            {
                                "text": best_text,
                                "original_text": AR_prediction,
                                "allsat": best_allsat,
                                "losses": best_losses,
                                "weighted_loss": best_weighted_loss,
                                "edited": True,
                            }
                        ],
                    }
                    
                    intermediate_output = {
                        "prompt": {
                            "text": source_text,
                        },
                        "generations": [
                            int_output
                        ],
                    }
                else:
                    output["generations"].append(
                        {
                                "text": best_text,
                                "original_text": AR_prediction,
                                "allsat": best_allsat,
                                "losses": best_losses,
                                "weighted_loss": best_weighted_loss,
                                "edited": True,
                        }
                    )
                    
                    intermediate_output["generations"].append(int_output)

                if sample_idx + 1 == config["num_samples"]:
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

    if (not interrupted):
        if config["task"] == "toxicity":
            # evaluate(run.path, outfile, 'toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n')
            evaluate(
                run.path,
                outfile,
                "toxicity-int,ppl-big,dist-n,repetition,fluency",
                toxicity_model_path=config["model_paths"][1],
                toxicity_model_type=config["model_types"][1],
            )  # 시간 문제로, perspective api 제외
        elif config["task"] == "formality":
            evaluate(
                run.path,
                outfile,
                "formality-int,formality-ext,ppl-big,dist-n,repetition,fluency",
                formality_model_path=config["model_paths"][1],
                formality_model_type=config["model_types"][1],
            )
        elif config["task"] == "sentiment":
            evaluate(
                run.path,
                outfile,
                "sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency",
                sentiment_model_path=config["model_paths"][1],
                sentiment_model_type=config["model_types"][1],
            )
        elif config["task"] == "sentiment-lewis-compr":
            evaluate(
                run.path,
                outfile,
                "sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency",
                sentiment_model_path=config["model_paths"][1],
                sentiment_model_type=config["model_types"][1],
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
        help="number of samples to edit per prompt",
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
    parser.add_argument("--closs_weight", type=float, default=0.32, help="closs weight")
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
