import yaml, os, torch, argparse, pickle, sys, json
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from laser_edit.utils.sc_energy_utils import load_sc_energy_dataset
from laser_edit.locate.ebm.locate_utils import LocateMachine4SCE
from laser_edit.locate.ebm.set_consistency_locate_instance import locate_ebm

sys.path.append("laser_edit/set_consistency_energy")
from energynets.energynet import energynet
from tasks.dataset_loader import concat_arbitrary_pairs
from energynets.decomposition.no_decomposition import no_decomposition_loader


        
def load_sc_energy_model_customized(config_path, device, locate_instance_type = ""):
    
    model_config = yaml.load(open(config_path), 
                                Loader=yaml.FullLoader)
    model_config['device'] = device

    if locate_instance_type != "":
        model_config['locate']['instance']['type'] = locate_instance_type

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
    
    return energy_net, model_config
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='eval2', choices=['eval2', 'test'])
    parser.add_argument('--use_incon_samples', action='store_true')
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ------------------------------------------------ #
    # Data Loading
    # ------------------------------------------------ #
    # 데이터셋을 정의한다.
    # load_sc_energy_dataset uses random.sample when n_samples is not None; default -1 means "all".
    n_samples_kw = None if args.n_samples < 0 else args.n_samples
    canonical_eval2_datasets, _ = load_sc_energy_dataset(
        args.dataset_name, args.split, args.use_incon_samples, n_samples_kw, args.random_seed
    )
    canonical_eval2_dataset = canonical_eval2_datasets[0]

    # ------------------------------------------------ #
    # Initialize EBM & Conduct Locate
    # ------------------------------------------------ #
    # Attention 모든 layer에 대해서, gradient norm에 대해서 총 13가지 케이스에 대해서 평가 실시한다.
    results = {}
    for locate_instance_type in ['attention', 'gradnorm']:
        
        energynet, locate_params = load_sc_energy_model_customized(args.params_path, args.device, locate_instance_type)
        locator = LocateMachine4SCE(locate_params, energynet, "lconvqa")

        # 데이터 로더를 정의한다.
        data_loader = no_decomposition_loader(canonical_eval2_dataset, tokenizer=energynet.representation_model.tokenizer, params=locate_params).get_loader()

        for agg_method in ['avg', 'median']:

            locator.params['locate']['instance']['agg_method'] = agg_method

            if locate_instance_type == 'attention':
                for num_layer in range(12):
                    
                    locator.params['locate']['instance']['attentions_num_layer'] = num_layer
                    
                    result = locate_ebm(locator, data_loader, device=locate_params['device'], params=None)

                    results[f"precision_{locate_instance_type}_{agg_method}_{num_layer}"] = result['precision']
                    results[f"recall_{locate_instance_type}_{agg_method}_{num_layer}"] = result['recall']
                    results[f"f1_{locate_instance_type}_{agg_method}_{num_layer}"] = result['f1']
                    results[f"accuracy_{locate_instance_type}_{agg_method}_{num_layer}"] = result['accuracy']

                    print(f'Type {locate_instance_type}, Agg {agg_method}, Layer {num_layer}: Precision = {result["precision"]}, Recall = {result["recall"]}, F1 = {result["f1"]}, Accuracy = {result["accuracy"]}')

            elif locate_instance_type == 'gradnorm':
                
                result = locate_ebm(locator, data_loader, device=locate_params['device'], params=None)

                results[f"precision_{locate_instance_type}_{agg_method}"] = result['precision']
                results[f"recall_{locate_instance_type}_{agg_method}"] = result['recall']
                results[f"f1_{locate_instance_type}_{agg_method}"] = result['f1']
                results[f"accuracy_{locate_instance_type}_{agg_method}"] = result['accuracy']

                print(f'Type {locate_instance_type}, Agg {agg_method}: Precision = {result["precision"]}, Recall = {result["recall"]}, F1 = {result["f1"]}, Accuracy = {result["accuracy"]}')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'locate_params_eval_result.jsonl'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()