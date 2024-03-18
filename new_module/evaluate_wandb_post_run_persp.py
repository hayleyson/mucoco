import argparse
import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluate_wandb import evaluate


def main(run_id_list, wandb_entity, wandb_project, prefix='mlm'):

    api = wandb.Api()
        
    for run_id in run_id_list:
        print("Doing run_id: ", run_id)
        
        run_path = f'{wandb_entity}/{wandb_project}/{run_id}'
        run = api.run(run_path)
        outfile_path_pattern_sub = None
        min_epsilon = run.config['min_epsilons'][0] if isinstance(run.config['min_epsilons'],list) else run.config['min_epsilons']
        if run_id == 'noe1aj78':
            outfile_path_pattern = "outputs/toxicity/mlm-reranking/mlm-token-nps4-k3-allsat_primary-0.5-0.5-wandb-2/outputs_epsilon-3-test.txt"
        elif run.config.get('output_dir_prefix', None):
            outfile_path_pattern = f"{run.config['output_dir_prefix']}/*{prefix}*{run_id}*/outputs_epsilon{min_epsilon}.txt"
        else:
            outfile_path_pattern = f"outputs/toxicity/mlm-reranking/**/*{prefix}*{run_id}*/outputs_epsilon{min_epsilon}.txt"
            outfile_path_pattern_sub = f"outputs/toxicity/mlm-reranking/*{prefix}*{run_id}*/outputs_epsilon{min_epsilon}.txt"
       
        
        print(outfile_path_pattern)
        outfile_paths = glob(outfile_path_pattern)
        if outfile_path_pattern_sub:
            outfile_paths += glob(outfile_path_pattern_sub)
        try:
            assert len(outfile_paths) == 1
        except:
            print(f"Warning. len(outfile_path) != 1: {len(outfile_paths)}. Skipping..")
            continue
        
        outfile_path = outfile_paths[0]    
        
        evaluate(run_path, outfile_path, 'toxicity')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id_list", nargs='+', type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="toxicity-decoding")
    args = parser.parse_args()
    main(args.run_id_list, wandb_entity="hayleyson", wandb_project=args.wandb_project, prefix="mlm")
        