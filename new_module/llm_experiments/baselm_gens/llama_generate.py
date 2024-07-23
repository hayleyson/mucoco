import joblib
import json
import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def generate_and_save_result(args):
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
    

    with open('/data/hyeryung/mucoco/new_module/data/dev_set.jsonl','r') as f:
        raw_data = f.readlines()
    prompts = [json.loads(line)['prompt']['text'] for line in raw_data]

    class CustomDataset(Dataset):
        def __init__(self, text_list):
            self.text_list = text_list
        
        def __len__(self):
            return len(self.text_list)
        
        def __getitem__(self, idx):
            return self.text_list[idx]
        
    class CollateFnClass():
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer       
            
        def collate_fn(self, input_batch):
                        
            return self.tokenizer(input_batch, truncation=True, padding=True, return_tensors="pt")

    myDataset = CustomDataset(prompts)
    myCollateFn = CollateFnClass(tokenizer)
    myDataLoader = DataLoader(myDataset, batch_size=1, collate_fn=myCollateFn.collate_fn)

    total_generated_result = []
    for batch in myDataLoader:
        batch = batch.to(device)
        generated_result = model.generate(**batch, 
                                        max_length=batch.input_ids.shape[-1] + 30,
                                        num_return_sequences=10,
                                        do_sample=True,
                                        top_p=0.96, 
                                        temperature=1.0)
        total_generated_result.append(generated_result[:, batch.input_ids.shape[-1]:])

    total_generated_text = []
    for result in total_generated_result:
        text_result = tokenizer.batch_decode(result)
        total_generated_text.append(text_result)
        
    formatted_generated_text_list = []

    for p, g in zip(prompts, total_generated_text):
        
        formatted_generated_text = {'prompt': {'text': p},
                                    'generations': [{'text': x} for x in g]}
        
        formatted_generated_text_list.append(json.dumps(formatted_generated_text))
        
    with open(args.file_save_path, 'w') as f:
        f.writelines([s + '\n' for s in formatted_generated_text_list])    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str)
    parser.add_argument('--file_save_path', type=str)
    
    args = parser.parse_args()
    
    generate_and_save_result(args)
    
    
    