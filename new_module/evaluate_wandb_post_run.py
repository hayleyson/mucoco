import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluate_wandb import evaluate


def main(run_id_list, wandb_entity, wandb_project,prefix="mlm"):

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
        del run
        
        if task == 'toxicity':
            # evaluate(run_path, outfile_path, 'toxicity-int,ppl-big,dist-n,repetition,fluency', 
            #          toxicity_model_path=model_path,toxicity_model_type=model_type) # 시간 문제로, perspective api 제외
            evaluate(run_path, outfile_path, 'toxicity-int,ppl-big,dist-n,repetition,fluency', 
                     toxicity_model_path=model_path,toxicity_model_type=model_type) # 시간 문제로, perspective api 제외
        elif task == 'formality':
            evaluate(run_path, outfile_path, 'formality-int,formality-ext,ppl-big,dist-n,repetition,fluency', 
                    formality_model_path=model_path,formality_model_type=model_type)
        elif task == 'sentiment':
            # evaluate(run_path, outfile_path, 'sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency',
            #         sentiment_model_path=model_path,sentiment_model_type=model_type)
            evaluate(run_path, outfile_path, 'sentiment-int,sentiment-ext,ppl-big,dist-n,repetition,fluency',
                    sentiment_model_path=model_path,sentiment_model_type=model_type)
    

if __name__ == "__main__":

    run_id_list = """5r9qmoze"""

    run_id_list = run_id_list.split('\n')
    print(run_id_list)
    main(run_id_list, wandb_entity="hayleyson", wandb_project="toxicity-decoding", prefix="mlm")
        