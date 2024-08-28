import joblib
import json
import argparse
import os
import time

import pandas as pd
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from new_module.llm_experiments.prompts import get_prompt
from new_module.evaluate_wandb import evaluate_main


def unravel(outputs_df):
    outputs_df=outputs_df.explode('generations',ignore_index=True)
    
    outputs_df['prompt']=outputs_df['prompt'].apply(lambda x: x['text'])
    
    outputs_df['text']=outputs_df['generations'].apply(lambda x: x['text'])
    
    gen_dict=outputs_df['generations'].values[0]
    
    for col in gen_dict.keys():
        outputs_df[col] = outputs_df['generations'].apply(lambda x: x.get(col,None))

    return outputs_df

def ravel(unraveled_df):
    if 'tokens' in unraveled_df:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text'],
                                                               'tokens': x['tokens']}],axis=1)
    else:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text']}],axis=1)
        
    return_df = unraveled_df.groupby('prompt')['generations'].sum([]).reset_index()
    return_df['prompt'] = return_df['prompt'].apply(lambda x: {'text': x})
    return return_df

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def generate_and_save_result(args):
    
    run = wandb.init(project="llm_experiments", entity="hayleyson", config=vars(args))
    
    device= "cuda" if torch.cuda.is_available() else "cpu"

    # Load model directly
    # Suppose you conducted huggingface-cli login and authenticated with your auth token
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    print(f'EOS token: {tokenizer.eos_token}')
    print(f'PAD token: {tokenizer.pad_token}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if '70b' in args.hf_model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_name, device_map="auto")
    else: 
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_name)
        model = model.to(device)
    

    with open(args.input_file_path,'r') as f:
        raw_data = f.readlines()
    if args.input_file_path.endswith('testset_gpt2_2500_locate.jsonl'):
        prompts = [json.loads(line)['prompt'] for line in raw_data]
        gens = [json.loads(line)['gen'] for line in raw_data]
    if args.input_file_path.endswith('testset_gpt2_2500.jsonl') or args.input_file_path.endswith('/data/hyeryung/mucoco/new_module/data/sentiment/dev_set.jsonl'):
        prompts_raw = [json.loads(line)['prompt']['text'] for line in raw_data]
        gens_raw = [json.loads(line)['generations'] for line in raw_data]
        gens = [] 
        prompts = []
        for i in range(len(gens_raw)):
            tmp_gen = [x['text'] for x in gens_raw[i]]
            gens.extend(tmp_gen)
            prompts.extend([prompts_raw[i]]*len(tmp_gen))
            
            


    class CustomDataset(Dataset):
        def __init__(self, text_list_1, text_list_2, system_prompt):
            self.text_list_1 = text_list_1
            self.text_list_2 = text_list_2
            self.system_prompt = system_prompt
        
        def __len__(self):
            return len(self.text_list_1)
        
        def __getitem__(self, idx):
            return self.system_prompt % (self.text_list_1[idx],self.text_list_2[idx])
        
    class CollateFnClass():
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer       
            
        def collate_fn(self, input_batch):
                        
            return self.tokenizer(input_batch, truncation=True, padding=True, return_tensors="pt"), input_batch 

    nontoxic_prompt = get_prompt(args)
    myDataset = CustomDataset(prompts, gens, nontoxic_prompt)
    myCollateFn = CollateFnClass(tokenizer)
    myDataLoader = DataLoader(myDataset, batch_size=1, collate_fn=myCollateFn.collate_fn)

    f = open(args.file_save_path, 'w')
    start_time = time.time()
    for i, (batch, batch_text) in enumerate(myDataLoader):
        batch = batch.to(device)
        generated_result = model.generate(**batch, 
                                        max_length=batch.input_ids.shape[-1] + args.max_tokens,
                                        num_return_sequences=args.num_return_sequences,
                                        do_sample=True,
                                        top_p=0.96, 
                                        temperature=1.0)
        
        text_result = tokenizer.batch_decode(generated_result, skip_special_tokens=True)
        print(text_result)
        total_generated_text = [x[len(batch_text[0]):] for x in text_result]
        print(total_generated_text)
        
        formatted_generated_text = {'prompt': {'text': prompts[i]},
                                    'initial_generations': {'text': gens[i]},
                                   'generations': [{'text': x} for x in total_generated_text]}
        
        f.write(json.dumps(formatted_generated_text) + '\n')
        f.flush()
    f.close()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    run.summary["execution_time"] = (end_time - start_time)
    run.finish()
    
    # if args.input_file_path.endswith('testset_gpt2_2500.jsonl'):
    #     data = pd.read_json(args.file_save_path, lines=True)
    #     data.to_json(args.file_save_path.replace('.jsonl','_bckp.jsonl'), orient='records', lines=True)
    #     data_1 = unravel(data)
    #     data_2 = ravel(data_1)
    #     data_2.to_json(args.file_save_path, orient='records', lines=True)
        
    # evaluate_main(
    #     run.path,
    #     args.file_save_path,
    #     "toxicity,ppl-big,dist-n,repetition,fluency,contents-preservation,qual",
    #     source_file_path=args.input_file_path,
    # )  # 시간 문제로, perspective api 제외

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str)
    parser.add_argument('--file_save_path', type=str)
    parser.add_argument('--input_file_path', type=str)
    parser.add_argument('--prompt_type', type=str)
    parser.add_argument('--num_return_sequences', type=int, default=10)
    parser.add_argument('--max_tokens', type=int, default=30)
    
    args = parser.parse_args()
    
    generate_and_save_result(args)
    
    
    