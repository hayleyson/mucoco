import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_wandb import evaluate_main

import os
os.chdir('/data/hyeryung/mucoco')

def main(run_id_list, wandb_entity, wandb_project,prefix="mlm"):

    api = wandb.Api()
        
    for run_id in run_id_list:
        print("Doing run_id: ", run_id)
        
        run_path = f'{wandb_entity}/{wandb_project}/{run_id}'
        run = api.run(run_path)
        outfile_path_pattern_sub = None
        min_epsilon = run.config['min_epsilons'][0] if isinstance(run.config['min_epsilons'],list) else run.config['min_epsilons']
        output_dir_prefix = run.config.get('output_dir_prefix', None)
        print(f"output_dir_prefix: {output_dir_prefix}")
        if output_dir_prefix is None:
            outfile_path_pattern = f"/data/hyeryung/mucoco/outputs/toxicity/mucola/*{prefix}*{run_id}*/outputs_epsilon{min_epsilon}.txt"
            # outfile_path_pattern = f"{run.config['output_dir_prefix']}/*{prefix}*{run_id}*/outputs_epsilon{min_epsilon}.txt"
        else:
            outfile_path_pattern = f"{output_dir_prefix}/*{run_id}*/outputs_epsilon{min_epsilon}.txt"
            # outfile_path_pattern_sub = f"outputs/toxicity/mlm-reranking/*{prefix}*{run_id}*/outputs_epsilon{min_epsilon}.txt"
        
        
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
        
        if run.config.get('model_paths', None) is not None:
            model_path = run.config['model_paths'][1]
            model_type = run.config['model_types'][1]
        else:
            model_path = run.config['model'].split(':')[1]
            model_type = run.config['model_types'].split(':')[1]
            
        if run.config.get('task', None) is not None:
            task = run.config['task']
        else:
            task = run.config['lossabbr'].split(':')[1]
        run.update()
        
        
        if task == 'toxicity':
            # metrics list: toxicity,toxicity-int,ppl-big,dist-n,repetition,fluency,contents-preservation,qual
            evaluate_main(run_path, outfile_path, 'toxicity-int,ppl-big,dist-n,repetition,fluency,contents-preservation', 
                     toxicity_model_path=model_path,toxicity_model_type=model_type,
                     source_file_path=run.config.get("source_data", None))
        elif task == 'formality':
            # metrics list: "formality-int,formality-ext,ppl-big,dist-n,repetition,fluency,contents-preservation,qual
            evaluate_main(run_path, outfile_path, 'contents-preservation,qual', 
                     formality_model_path=model_path,formality_model_type=model_type,
                     source_file_path='') ## source_file_path is set within evaluate_wandb
        elif task == 'sentiment':
            # metrics list: sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency,contents-preservation,qual
            evaluate_main(run_path, outfile_path, 'sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency,contents-preservation,qual', 
                     sentiment_model_path=model_path,sentiment_model_type=model_type,
                     source_file_path=run.config.get("source_data", None))
        del run

if __name__ == "__main__":

    run_id_list = """q7tlrfcl
r7kykwge"""

    run_id_list = run_id_list.split('\n')
    main(run_id_list, wandb_entity="hayleyson", wandb_project="toxicity-decoding", prefix="")
            