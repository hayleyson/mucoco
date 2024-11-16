import argparse
import os
import sys
from glob import glob

import pandas as pd
from transformers import AutoTokenizer

import wandb
from new_module.evaluation.evaluate_wandb import evaluate

## toxicity
origin=pd.read_json('new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl', lines=True)
origin=origin.explode('generations')
origin['text']=origin['generations'].apply(lambda x: x['text'])
gen_dict=origin['generations'].values[0]
for col in gen_dict.keys():
    origin[col] = origin['generations'].apply(lambda x: x.get(col, None))
origin=origin.reset_index(drop=True)
origin=origin.rename(columns={'text':'origin_text', 'tokens':'origin_tokens'})
origin_toxicity=origin

## sentiment
origin=pd.read_json('new_module/data/sentiment/outputs.txt.init.jsonl', lines=True)
origin=origin.explode('generations')
origin['text']=origin['generations'].apply(lambda x: x['text'])
gen_dict=origin['generations'].values[0]
for col in gen_dict.keys():
    origin[col] = origin['generations'].apply(lambda x: x.get(col, None))
origin=origin.reset_index(drop=True)
origin=origin.rename(columns={'text':'origin_text', 'tokens':'origin_tokens'})
origin_sentiment=origin

## formality
with open('data/formality/GYAFC_Corpus/Entertainment_Music/test/informal', 'r') as f:
    origin = [line.rstrip('\n') for line in f.readlines()]
origin=pd.DataFrame({'origin_text':origin})
tokenizer=AutoTokenizer.from_pretrained('/shared/s3/lab07/hyeryung/loc_edit/roberta-base-pt16-formality-classifier-with-gpt2-large-embeds/step_1116_best_checkpoint/')
origin['origin_tokens']=origin['origin_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=False))
origin_formality=origin

def create_fillin_outputs(run_id_list, wandb_entity, wandb_project,prefix="mlm"):

    global origin_toxicity, origin_sentiment, origin_formality

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
            
        if run.config.get('task', None) is not None:
            task = run.config['task']
        else:
            task = run.config['lossabbr'].split(':')[1]
        run.update()
        del run
        
        if task == 'toxicity':
            origin=origin_toxicity            
        elif task == 'formality':
            origin=origin_formality
        elif task == 'sentiment':
            origin = origin_sentiment
            
        outputs=pd.read_json(outfile_path,lines=True)
        outputs0 = outputs.copy()
        outputs0['prompt1']=outputs0['prompt'].apply(lambda x: x['text'])
        
        outputs=outputs.explode('generations')
        outputs['prompt1']=outputs['prompt'].apply(lambda x: x['text'])
        outputs['text']=outputs['generations'].apply(lambda x: x['text'])
        gen_dict=outputs['generations'].values[0]
        for col in gen_dict.keys():
            outputs[col] = outputs['generations'].apply(lambda x: x.get(col, None))
        outputs=outputs.reset_index(drop=True)
        
        outputs=pd.concat([outputs, origin[['origin_text', 'origin_tokens']]], axis=1)
        outputs.loc[outputs['weighted_loss']==-1,'text']=outputs.loc[outputs['weighted_loss']==-1,'origin_text']
        outputs.loc[outputs['weighted_loss']==-1,'tokens']=outputs.loc[outputs['weighted_loss']==-1,'origin_tokens']
        outputs['generations1']=None
        
        for i, row in outputs.iterrows():
            outputs.loc[i, 'generations1'] = [{'text': row['text'],
                                'tokens': row['tokens'],
                                'indices': row['indices'],
                                'orig_tokens_at_indices': row['orig_tokens_at_indices'],
                                'allsat': row['allsat'],
                                'losses': row['losses'],
                                'weighted_loss': row['weighted_loss']}]
        outputs = outputs.groupby('prompt1')['generations1'].agg(lambda x: sum(x, [])).reset_index()
        
        outputs=pd.merge(outputs, outputs0[['prompt','prompt1']],on='prompt1',how='left')
        assert outputs.shape[0] == outputs0.shape[0]
        outputs = outputs.rename(columns={'generations1': 'generations'})
        outputs = outputs[['prompt', 'generations']].copy()
        outputs.to_json(os.path.splitext(outfile_path)[0]+'_filled.txt',
                            lines=True, orient='records')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id_list", type=str, default=None, help="\\n separated list of wandb run ids")
    parser.add_argument("--wandb_entity", type=str, default="hayleyson")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="mlm")
    args = parser.parse_args()

    run_id_list = args.run_id_list.split('\n')
    create_fillin_outputs(run_id_list, wandb_entity=args.wandb_entity, wandb_project=args.wandb_project, prefix=args.prefix)
        