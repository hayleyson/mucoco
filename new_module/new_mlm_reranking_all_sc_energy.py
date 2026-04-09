
###########################################################
# Package import 
import joblib
import argparse
import os
import yaml
import random
from dataclasses import dataclass
import json
import math
from copy import deepcopy
import time
import logging

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
import wandb

import new_module.losses as lossbuilder
from new_module.evaluation.evaluate_pipeline import run_generation_evaluation
from new_module.locate.new_locate_utils import LocateMachine4SCE
from new_module.set_consistency_energy.energynets.energynet import energynet
from new_module.new_decode_utils import analyze_span_lengths_and_count, editing_4sce

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGGING_LEVEL", logging.DEBUG))

random.seed(42)

# Set global variables
root_dir = 'new_module/set_consistency_energy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###########################################################

def main(config):

    ###########################################################
    # Initialize wandb & output files
    ###########################################################
    if not config["debug"]:
        run = wandb.init(
                    project=config["wandb_project"],
                    entity=config["wandb_entity"],
                    config=config,
                )

        run_id = str(run.path.split("/")[-1])
        outdir = os.path.join(config['output_dir_prefix'], run_id)
    else:
        outdir = os.path.join(config['output_dir_prefix'], 'debug')
    
    os.makedirs(outdir, exist_ok=True)
    
    outfile = f"{outdir}/outputs.txt"
    outf = open(outfile, "w")
    int_outf = open(outfile+".intermediate", "w")


    ###########################################################
    # Load models 
    ###########################################################
    
    # 1) MLM
    mlm = AutoModelForMaskedLM.from_pretrained(config["mlm_path"])
    mlm.eval()
    mlm.to(device)
    mlm_tokenizer = AutoTokenizer.from_pretrained(config["mlm_path"])

    # 2) Causal LM
    causal_lm = AutoModelForCausalLM.from_pretrained(config["causal_lm_path"])
    causal_lm.eval()
    causal_lm.half()
    causal_lm.to(device)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(config["causal_lm_path"])
    causal_lm_tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})

    # 3) Energy Net
    model_config = yaml.load(open(config["ebm_params_path"]), 
                                Loader=yaml.FullLoader)
    model_config['device'] = device

    energy_net = energynet(params=model_config)
    energy_net.load_state_dict(torch.load(model_config["model_path"], 
                                        map_location=model_config['device'],
                                        weights_only=True)['state_dict'], strict=False)
    if 'threshold' in torch.load(model_config["model_path"],
                                map_location=model_config['device'],
                                weights_only=True):
        energy_net.threshold = torch.load(model_config["model_path"],
                                map_location=model_config['device'],
                                weights_only=True)['threshold']
    energy_net.eval()
    energy_net.to(device)

    energy_net_tokenizer = energy_net.representation_model.tokenizer
    energy_net_tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})

    # log locate method to wandb
    if not config["debug"]:
        wandb.log({"locate_type_inst": model_config["locate"]["instance"]["type"]})
        wandb.log({"locate_type_span": model_config["locate"]["span"]["type"]})

    ###########################################################
    # Wrap models into loss functions
    ###########################################################

    @dataclass
    class LossArgs:
        length_normalize: bool = True
        alpha: float = 1.0
        AR_temperature: float = 1.0
        AR_top_k: int = 0
        AR_top_p: float = 0.96
        max_output_length: int = 20
        task: str = None
        device: str = None

    loss_args = LossArgs(task=task,
                        device=device)

    lossfns = []
    losses = config['losses']
    models = [causal_lm, energy_net]
    tokenizers = [causal_lm_tokenizer, energy_net_tokenizer]

    for i, loss in enumerate(losses):
        lossfns.append(
            lossbuilder.build_loss(
                loss,
                models[i],
                tokenizers[i],
                loss_args,
            )
        )

    ###########################################################
    # Set up min_epsilons
    ###########################################################

    # if min_epsilon is -1, set it to the threshold of energy net (default: -1)
    if config['min_epsilons'][0] == -1:
        config['min_epsilons'][0] = lossfns[1].model.threshold

    ###########################################################
    # Set up LocateMachine4SCE
    ###########################################################

    locator = LocateMachine4SCE(model_config, energy_net, task)

    ###########################################################
    # Load dataset
    ###########################################################

    with open(config['source_data_path'], 'r') as f:
        data = [json.loads(line.rstrip()) for line in f]

        
        
    ###########################################################
    # Main logic
    ###########################################################
    
    decode_start_time = time.time()
    num_skipped = 0
    num_edited = 0
    num_decoded_tokens = 0

    # NOTE. batch_size = 1
    for i in range(len(data)):
        
        logger.debug(f"================================ Doing {i}th sample ==================================")

        source_text = data[i]['prompt']['text']
        AR_prediction_all = [data[i]['generations'][0]['text']]
        
        curr_loss = torch.zeros(len(AR_prediction_all)).to(config['device'])
        logging_loss = torch.zeros((len(AR_prediction_all),len(config["losses"]))).to(config['device'])
        edit_yn = torch.ones(len(AR_prediction_all), dtype=torch.bool).to(config['device'])
                
        for lossid, _ in enumerate(config["losses"]):
            with torch.no_grad():
                lossvalue = lossfns[lossid].compute_gold_loss(
                    source_text, AR_prediction_all,
                    label_id=config['target_label_ids'][lossid],
                )
                torch.cuda.empty_cache()
            curr_loss += config["loss_weights"][lossid] * lossvalue
            logging_loss[:, lossid] = lossvalue.clone()

        
        allsat = logging_loss[:,1] <= config['min_epsilons'][0]
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

        # If the sample satisfies constraints and dont_skip_allsat is not passed, skip the sample (Recall: batch size = 1)
        if (edit_yn.sum().item() == 0) and (not config["dont_skip_allsat"]):
            
            num_edited += 0
            num_skipped += len(AR_prediction_all)
            num_decoded_tokens += 0
            
            
            logger.info(
                    f"skipping this sample since it already satisfies constraint. {best_losses}"
                )
        
        else:
            
            num_edited += edit_yn.sum().item()
            num_skipped += (len(AR_prediction_all) - edit_yn.sum().item())
            num_decoded_tokens += sum([len(x) for x in causal_lm_tokenizer(running_text).input_ids])       
            located_instance_list = []
        
            for _iter in range(config['n_iter']):
                
                logger.debug(f"!!! {_iter}th iteration !!!")
                
                ###########################################################
                # Locate
                ###########################################################
                if model_config["locate"]["span"]["type"] == "ground_truth": # default: ground_truth
                    masked_text, prediction_list = locator.locate_main_with_gt_span(running_text, max_num_tokens=config["num_edit_tokens_per_step"], unit='word')
                else:
                    masked_text, prediction_list = locator.locate_main(running_text, max_num_tokens=config["num_edit_tokens_per_step"], unit='word')
                
                if _iter == 0:
                    located_instance_list = prediction_list
                else:
                    for index, item in enumerate(located_instance_list):
                        located_instance_list[index].extend(prediction_list[index])
                

                ###########################################################
                # Edit
                ###########################################################
                
                _, span_lengths = analyze_span_lengths_and_count(masked_text[0])
                    
                final_hypotheses_ = []
                new_best_weighted_loss_ = []
                new_best_allsat_ = []
                new_best_logging_loss_ = []
                
                tmp_masked_text = []
                tmp_running_text = []
                edit_ixes_before_marking = edit_yn.nonzero().squeeze(-1)
                
                if len(span_lengths) > 0:
                    test_sent = masked_text[0]
                    
                    final_hypotheses_curr, new_best_weighted_loss_curr, new_best_allsat_curr, new_best_logging_loss_curr = \
                                editing_4sce(source_text, 
                                    running_text[0], 
                                    masked_text[0], 
                                    span_lengths,
                                    prediction_list[0][0],
                                    mlm, 
                                    mlm_tokenizer, 
                                    lossfns, 
                                    config, 
                                    batch_size=32,
                                    post_context_mode="original")
                                
                    final_hypotheses_.extend(final_hypotheses_curr)
                    new_best_weighted_loss_.append(new_best_weighted_loss_curr)
                    new_best_allsat_.append(new_best_allsat_curr)
                    new_best_logging_loss_.append(new_best_logging_loss_curr)
                    tmp_masked_text.append(test_sent)
                    tmp_running_text.append(running_text[0])
                                
                    if len(new_best_weighted_loss_) == 0:
                        new_best_weighted_loss_ = torch.empty((0,)).to(config['device'])
                        new_best_allsat_ = torch.empty((0,)).bool().to(config['device'])
                        new_best_logging_loss_ = torch.empty((0, len(config['losses']))).to(config['device'])
                    else:
                        new_best_weighted_loss_ = torch.cat(new_best_weighted_loss_)
                        new_best_allsat_ = torch.cat(new_best_allsat_)
                        new_best_logging_loss_ = torch.cat(new_best_logging_loss_, dim=0)
                    

                    # Variables that end with "_" are tensors with the same length as running_text
                    # But we want variables without "_" suffix to be tensors with the same length as AR_prediction_all
                    # Thus, we first declare tensors with the same length as AR_prediction_all, and then update the values for the edit_yn indices.
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
                    update = (update & edit_yn) # edit 대상인 것들만 update하기 위해서 update 조건에 edit_yn을 sum.

                    ## intermediate output for debugging
                    for sample_ix in range(len(running_text)): # edit 대상인 것들만 update.
                        int_output[edit_ixes[sample_ix]].update({f"iter{_iter}_original_sentence": running_text[sample_ix],
                                                                f"iter{_iter}_masked_sentence": masked_text[sample_ix],
                                                                f"iter{_iter}_best_text": final_hypotheses[edit_ixes[sample_ix]],
                                                                f"iter{_iter}_update": update[edit_ixes[sample_ix]].item(),
                                                                f"iter{_iter}_located_instance": prediction_list[edit_ixes[sample_ix]][0]})    
                    
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
            

                    
              
                else:
                    
                    logger.warning(
                            f"No span is detected during locate step. Skipping this sample. Text: {running_text[0]}"
                        )
                    break
                    
                    
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
                    "located_instances": located_instance_list[i],
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
                    
    outf.close()
    int_outf.close()

    if not config["debug"]:
        run.summary["decode_time"] = time.time() - decode_start_time
        run.summary['num_decoded_tokens'] = num_decoded_tokens
        run.summary['toks_p_sec'] = (num_decoded_tokens/run.summary['decode_time'])
        run.summary["num_skipped"] = num_skipped
        run.summary["num_edited"] = num_edited
        run.summary["outfile_path"] = outfile

        run.finish()
    else:
        decode_time = time.time() - decode_start_time
        logger.info(f"decode_time: {decode_time}")
        logger.info(f"num_skipped: {num_skipped}")
        logger.info(f"num_edited: {num_edited}")  
        logger.info(f"nun_decoded_tokens: {num_decoded_tokens}")
        logger.info(f"toks_p_sec: {num_decoded_tokens/decode_time}")
    
    run_generation_evaluation(
            "",
            outfile,
            "set-consistency,set-consistency-gpt,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
            source_file_path=config["source_data_path"],
            task=task,
        )  
        
if __name__ == "__main__":
    
    

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("source_data_path", type=str)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--losses", nargs="+", type=str, default=['gpt2_no_prefix', 'sc_energy'])
    parser.add_argument("--min_epsilons", nargs="+", type=float, default=[-1], help="not used for sc_energy")
    parser.add_argument("--loss_weights", nargs="+", type=float, default=[1.0, 10.0])
    parser.add_argument("--k_per_location", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=4)
    parser.add_argument("--selection_criteria", type=str, choices=["weighted_sum", "allsat_primary"], default="allsat_primary",)
    parser.add_argument("--slurm_job_id", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--ebm_params_path", type=str, help="Path to model-specific YAML configuration")
    parser.add_argument("--causal_lm_path", type=str, default="gpt2-large")
    parser.add_argument("--mlm_path", type=str, default="roberta-base")
    parser.add_argument("--dont_skip_allsat", action="store_true", help="if this argument is passed, the module will conduct decoding on all samples even if they already satisfy constraints",)
    parser.add_argument("--num_edit_tokens_per_step", type=int, default=2)
    parser.add_argument("--max_tokens_per_span", type=int, default=2)
    args = parser.parse_args()


    main_start_time = time.time()

    task = args.task
    config = vars(args)
    config.update({'task': task, 
            'device': device,
            'target_label_ids': [1, 1],
            'consider_prompt_for_cand_gen': False,
            'output_dir_prefix': f'outputs/sc_energy/{task}/ebm/',
            })

    ###########################################################
    # Main
    ###########################################################
    
    main(config)

