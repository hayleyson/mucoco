import os
import json
import argparse
import time

from tqdm import tqdm
from openai import OpenAI
import pandas as pd

from new_module.llm_experiments.locate_with_llm.prompts import get_prompt

def run_api_and_save_result(dataset, args):
    
    start_time = time.time()
    file_save_path = f"new_module/llm_experiments/locate_with_llm/results/{args.model}_{args.prompt_type}_{args.dataset}_{str(int(start_time))}.jsonl"
    execution_time_path = file_save_path.replace(".jsonl", ".time")
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    
    if 'cot' in args.prompt_type:
        schema = {
                    "name": "span_list_with_reasoning",
                    "type": "json_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "spans": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "reasoning": {
                                "type": "string",
                            }
                        },
                        "required": ["reasoning", "spans"],
                        "additionalProperties": False
                    }
                }
        
    else:
        schema = {
            "name": "span_list",
            "type": "json_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["spans"],
                "additionalProperties": False
            }
        }
        
    print('-'*100)
    print(schema)
    
    system_prompt, user_prompt = get_prompt(args)
    print('-'*100)
    print(f"prompt_type: {args.prompt_type}")
    print(f"system_prompt: {system_prompt}")
    print(f"user_prompt: {user_prompt}")
    
    if args.dataset == "toxicspans_extended":
        prompts = [""] * len(dataset)
        generations = dataset['text'].tolist()
    else:
        prompts = dataset['prompt'].tolist()
        generations = dataset['generation'].tolist()
        
    ## generate responses
    responses = []
    f = open(file_save_path, 'w')
    max_prompt_count = len(prompts) if args.num_test_prompts == -1 else args.num_test_prompts
    
    if "o3" in args.model or "o4" in args.model or "gpt-5" in args.model: # reasoning models
        for p, g in tqdm(zip(prompts[:max_prompt_count], generations[:max_prompt_count])):
            
            response = client.responses.create(
                model = args.model,
                reasoning = {"effort": "medium"},
                max_output_tokens = args.max_tokens,
                input = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (user_prompt % (p, g)) if p != "" else (user_prompt % g)
                    }
                ],
                text = {
                    "format": schema
                },    
            )
            responses.append(response)
            
            print(response)
            
            # For reasoning models, the output structure is:
            # output[0] = ResponseReasoningItem (metadata, content=None)
            # output[1] = ResponseOutputMessage (actual text output)
            # We need to find the ResponseOutputMessage item, not the ResponseReasoningItem
            
            if response.status == "incomplete":
                print(f"Warning: Response incomplete - {response.incomplete_details.reason if response.incomplete_details else 'unknown reason'}")
                formatted_generated_text = ""  # will not save even partial answer since we expect json-serializable output
            else:
                # Find the output message (not the reasoning item)
                formatted_generated_text = response.output_text
            
            f.write(json.dumps(formatted_generated_text) + '\n')
            f.flush() 
            
    else:
        for p, g in tqdm(zip(prompts[:max_prompt_count], generations[:max_prompt_count])):
            
            response = client.responses.create(
                model = args.model,
                top_p = args.top_p,
                temperature = args.temperature,
                max_output_tokens = args.max_tokens,
                input = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (user_prompt % (p, g)) if p != "" else (user_prompt % g)
                    }
                ],
                text = {
                    "format": schema
                },    
            )
            responses.append(response)
            print(response)
            
            # formatted_generated_text = response.output[0].content[0].text
            formatted_generated_text = response.output_text
    
            f.write(json.dumps(formatted_generated_text) + '\n')
            f.flush() 
    f.close()
    end_time = time.time()
    
    with open(execution_time_path, 'w') as f:
        f.write(str(end_time - start_time) + "\n")
        
    print(f"Total time taken: {end_time - start_time}")


def run_whitebox_and_save_result(args):
    
    # device= "cuda" if torch.cuda.is_available() else "cpu"

    # # Load model directly
    # # Suppose you conducted huggingface-cli login and authenticated with your auth token
    # tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    # print(f'EOS token: {tokenizer.eos_token}')
    # print(f'PAD token: {tokenizer.pad_token}')
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    
    # if '70b' in args.hf_model_name.lower():
    #     model = AutoModelForCausalLM.from_pretrained(args.hf_model_name, device_map="auto")
    # else: 
    #     model = AutoModelForCausalLM.from_pretrained(args.hf_model_name)
    #     model = model.to(device)
    

    # with open(args.input_file_path,'r') as f:
    #     raw_data = f.readlines()
    # if args.input_file_path.endswith('jsonl'): ##toxic,senti
    #     prompts = [json.loads(line)['prompt']['text'] for line in raw_data]
    # else:## txt file ##formality transfer
    #     prompts = [line.rstrip() for line in raw_data]


    # class CustomDataset(Dataset):
    #     def __init__(self, text_list, system_prompt):
    #         self.text_list = text_list
    #         self.system_prompt = system_prompt
        
    #     def __len__(self):
    #         return len(self.text_list)
        
    #     def __getitem__(self, idx):
    #         return self.system_prompt % self.text_list[idx]
        
    # class CollateFnClass():
    #     def __init__(self, tokenizer):
    #         self.tokenizer = tokenizer       
            
    #     def collate_fn(self, input_batch):
                        
    #         return self.tokenizer(input_batch, truncation=True, padding=True, return_tensors="pt"), input_batch 

    # nontoxic_prompt = get_prompt(args)
    # myDataset = CustomDataset(prompts, nontoxic_prompt)
    # myCollateFn = CollateFnClass(tokenizer)
    # myDataLoader = DataLoader(myDataset, batch_size=1, collate_fn=myCollateFn.collate_fn)

    # f = open(args.file_save_path, 'w')
    # start_time = time.time()
    # for prompt, (batch, batch_text) in zip(prompts, myDataLoader):
    #     batch = batch.to(device)
    #     generated_result = model.generate(**batch, 
    #                                     max_length=batch.input_ids.shape[-1] + args.max_tokens,
    #                                     num_return_sequences=args.num_return_sequences,
    #                                     do_sample=True,
    #                                     top_p=0.96, 
    #                                     temperature=1.0)
        
    #     text_result = tokenizer.batch_decode(generated_result, skip_special_tokens=True)
    #     total_generated_text = [x[len(batch_text[0]):] for x in text_result]
        
    #     formatted_generated_text = {'prompt': {'text': prompt},
    #                                 'generations': [{'text': x} for x in total_generated_text]}
        
    #     f.write(json.dumps(formatted_generated_text) + '\n')
    #     f.flush()
    # f.close()
    # end_time = time.time()
    # print(f"Total time taken: {end_time - start_time}")
    pass
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('prompt_type', type=str)
    parser.add_argument('dataset', type=str, choices=["toxicspans", "toxicspans_extended", "inconsistentspans"])
    parser.add_argument('--num_test_prompts', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=30)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--do_sample', type=bool, default=False, help='Deprecated. Use temperature instead.')
    args = parser.parse_args()

    # load data
    
    print(os.getcwd())
    
    if args.dataset == "toxicspans":
        dataset = pd.read_json("new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl", lines=True)
    
    elif args.dataset == "toxicspans_extended":
        dataset = pd.read_json("new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl", lines=True)
    
    elif args.dataset == "inconsistentspans":
        dataset = pd.read_json("new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl", lines=True)
        dataset = dataset.rename(columns = {"premise": "prompt",
                                  "hypothesis": "generation"},)
    

    # run locate 
    
    if ("gpt" in args.model) or ("o3" in args.model) or ("o4" in args.model): 
        run_api_and_save_result(dataset, args)
        
    elif "llama" in args.model:
        run_whitebox_and_save_result(dataset, args)
    
        




