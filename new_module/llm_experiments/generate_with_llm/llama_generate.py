import joblib
import json
import argparse
import os
import time

import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from new_module.llm_experiments.prompts import get_prompt


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
    if args.input_file_path.endswith('jsonl'): ##toxic,senti
        prompts = [json.loads(line)['prompt']['text'] for line in raw_data]
    else:## txt file ##formality transfer
        prompts = [line.rstrip() for line in raw_data]


    class CustomDataset(Dataset):
        def __init__(self, text_list, system_prompt):
            self.text_list = text_list
            self.system_prompt = system_prompt
        
        def __len__(self):
            return len(self.text_list)
        
        def __getitem__(self, idx):
            return self.system_prompt % self.text_list[idx]
        
    class CollateFnClass():
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer       
            
        def collate_fn(self, input_batch):
                        
            return self.tokenizer(input_batch, truncation=True, padding=True, return_tensors="pt"), input_batch 

    nontoxic_prompt = get_prompt(args)
    myDataset = CustomDataset(prompts, nontoxic_prompt)
    myCollateFn = CollateFnClass(tokenizer)
    myDataLoader = DataLoader(myDataset, batch_size=1, collate_fn=myCollateFn.collate_fn)

    f = open(args.file_save_path, 'w')
    start_time = time.time()
    for prompt, (batch, batch_text) in zip(prompts, myDataLoader):
        batch = batch.to(device)
        generated_result = model.generate(**batch, 
                                        max_length=batch.input_ids.shape[-1] + args.max_tokens,
                                        num_return_sequences=args.num_return_sequences,
                                        do_sample=True,
                                        top_p=0.96, 
                                        temperature=1.0)
        
        text_result = tokenizer.batch_decode(generated_result, skip_special_tokens=True)
        total_generated_text = [x[len(batch_text[0]):] for x in text_result]
        
        formatted_generated_text = {'prompt': {'text': prompt},
                                    'generations': [{'text': x} for x in total_generated_text]}
        
        f.write(json.dumps(formatted_generated_text) + '\n')
        f.flush()
    f.close()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    run.summary["execution_time"] = (end_time - start_time)
    run.finish()
    
    
    

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
    
    
    