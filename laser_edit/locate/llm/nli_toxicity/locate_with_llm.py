import os
import json
import argparse
import time
import importlib.util
import sys
import dotenv
    
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import torch

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    LLM = None
    SamplingParams = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

from laser_edit.locate.llm.nli_toxicity.prompts import get_prompt

dotenv.load_dotenv()

def get_bbm_prompts(dataset_name, prompt_type):
    """Load BBM prompts from external files."""
    prompt_file = f"laser_edit/data/BIG-Bench-Mistake/mistake_finding_prompts/{dataset_name}_prompts.py"
    spec = importlib.util.spec_from_file_location("bbm_prompts", prompt_file)
    bbm_prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bbm_prompts)
    
    if prompt_type == "direct":
        messages_base = list(bbm_prompts.DIRECT_MESSAGES)
        template = bbm_prompts.DIRECT_TEMPLATE
    elif prompt_type == "per-step":
        messages_base = list(bbm_prompts.DIRECT_PERSTEP_MESSAGES)
        template = bbm_prompts.DIRECT_PERSTEP_TEMPLATE
    elif prompt_type == "cot":
        messages_base = list(bbm_prompts.COT_PERSTEP_MESSAGES)
        template = bbm_prompts.COT_PERSTEP_TEMPLATE
    else:
        raise ValueError(f"Unsupported prompt_type for BBM: {prompt_type}")
        
    return messages_base, template

def format_bbm_steps(steps_list):
    """Format BBM steps into 'Thought N: ...' strings."""
    formatted_steps = []
    for steps in steps_list:
        formatted = ""
        for i, step in enumerate(steps):
            formatted += f"Thought {i}: {step}\n"
        formatted_steps.append(formatted.strip())
    return formatted_steps

def run_api_and_save_result(dataset, args):
    
    start_time = time.time()
    file_save_path = f"laser_edit/locate/llm/nli_toxicity/results/{args.model}_{args.prompt_type}_{args.dataset}_{str(int(start_time))}.jsonl"
    execution_time_path = file_save_path.replace(".jsonl", ".time")
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    
    
    is_bbm = args.dataset in ["logical_deduction", "tracking_shuffled_objects"]
    
    if is_bbm:
        messages_base, template = get_bbm_prompts(args.dataset, args.prompt_type)
        schema = {
            "name": "bbm_answer",
            "type": "json_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The thought number where the first mistake occurs, or 'No' if no mistake."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Step-by-step reasoning."
                    }
                },
                "required": ["answer", "reasoning"],
                "additionalProperties": False
            }
        }
    else:
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
    
    if not is_bbm:
        system_prompt, user_prompt = get_prompt(args)
        print('-'*100)
        print(f"prompt_type: {args.prompt_type}")
        print(f"system_prompt: {system_prompt}")
        print(f"user_prompt: {user_prompt}")
    
    if is_bbm:
        prompts = dataset['input'].tolist()
        generations = format_bbm_steps(dataset['steps'].tolist())
    elif args.dataset == "toxicspans_extended":
        prompts = [""] * len(dataset)
        generations = dataset['text'].tolist()
    else:
        prompts = dataset['prompt'].tolist()
        generations = dataset['generation'].tolist()
        
    ## generate responses
    responses = []
    f = open(file_save_path, 'w')
    max_prompt_count = len(prompts) if args.num_test_prompts == -1 else args.num_test_prompts
    
    if "o3" in args.model or "o4" in args.model or "gpt-5" in args.model or ("o1" in args.model.lower()): # reasoning models
        for p, g in tqdm(zip(prompts[:max_prompt_count], generations[:max_prompt_count])):
            
            response = client.responses.create(
                model = args.model,
                reasoning = {"effort": args.reasoning_effort},
                max_output_tokens = args.max_tokens,
                input = messages_base + [{"role": "user", "content": template.format(input=p, steps=g)}] if is_bbm else [
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
                input = messages_base + [{"role": "user", "content": template.format(input=p, steps=g)}] if is_bbm else [
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


def run_whitebox_and_save_result(dataset, args):
    if LLM is None:
        raise ImportError("vllm or transformers not found. Please ensure they are installed.")

    start_time = time.time()
    file_dir = "laser_edit/locate/llm/nli_toxicity/results/"
    os.makedirs(file_dir, exist_ok=True)
    file_save_name = f"{args.model.split('/')[-1]}_{args.prompt_type}_{args.dataset}_{str(int(start_time))}.jsonl"
    file_save_path = os.path.join(file_dir, file_save_name)
    execution_time_path = file_save_path.replace(".jsonl", ".time")
    print(f"Saving results to {file_save_path}")

    is_bbm = args.dataset in ["logical_deduction", "tracking_shuffled_objects"]
    
    if is_bbm:
        messages_base, template = get_bbm_prompts(args.dataset, args.prompt_type)
    else:
        system_prompt, user_prompt = get_prompt(args)
    
    if args.use_vllm:
        print(f"Initializing vLLM with {args.model}...")
        llm = LLM(model=args.model, gpu_memory_utilization=0.8)
        if "Qwen3-8B" in args.model:
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=args.max_tokens)
        else:
            sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if is_bbm:
            prompts_raw = dataset['input'].tolist()
            generations_raw = format_bbm_steps(dataset['steps'].tolist())
        elif args.dataset == "toxicspans_extended":
            prompts_raw = [""] * len(dataset)
            generations_raw = dataset['text'].tolist()
        else:
            prompts_raw = dataset['prompt'].tolist()
            generations_raw = dataset['generation'].tolist()

        max_prompt_count = len(prompts_raw) if args.num_test_prompts == -1 else args.num_test_prompts
        
        inputs_for_vllm = []
        for p, g in zip(prompts_raw[:max_prompt_count], generations_raw[:max_prompt_count]):
            if is_bbm:
                user_content = template.format(input=p, steps=g)
                messages = messages_base + [{"role": "user", "content": user_content}]
            else:
                user_content = (user_prompt % (p, g)) if p != "" else (user_prompt % g)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                
            if "Qwen3-8B" in args.model:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            else:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs_for_vllm.append(prompt_text)

        print("Running vLLM inference...")
        outputs = llm.generate(inputs_for_vllm, sampling_params)
        
        num_written = 0
        with open(file_save_path, 'w') as f:
            for output in outputs:
                generated_text_raw = output.outputs[0].text
                
                if args.model.split('/')[-1] == 'Qwen3-8B':
                    # 1. Extract content after thinking process
                    if '</think>' in generated_text_raw:
                        content = generated_text_raw.split('</think>')[-1].strip()
                    elif r"<\/think>" in generated_text_raw:
                        content = generated_text_raw.split(r"<\/think>")[-1].strip()
                    else:
                        content = None # Truncated/Invalid

                    # 2. Format based on task and success
                    if content is not None:
                        if is_bbm:
                            try:
                                parsed = json.loads(content)
                                if isinstance(parsed, dict):
                                    generated_text = parsed
                                else:
                                    generated_text = {"answer": str(parsed)}
                            except:
                                generated_text = {"answer": content}
                        else:
                            try:
                                generated_text = json.loads(content)
                            except:
                                generated_text = {"spans": [], "error": "failed to parse json"}
                    else:
                        # 3. Handle truncated responses
                        if is_bbm:
                            generated_text = {"answer": "error: truncated", "reasoning": "truncated before </think>"}
                        else:
                            generated_text = {"spans": [], "error": "truncated before </think>"}
                    f.write(json.dumps(generated_text) + '\n')
                    num_written += 1

                elif args.model.split('/')[-1] == 'QwQ-32B-Preview':

                    f.write(generated_text_raw + '\n')
                    num_written += 1

                else:
                    f.write(json.dumps(generated_text_raw) + '\n')
                    num_written += 1
                
        end_time = time.time()
        f.close()
        with open(execution_time_path, 'w') as f:
            f.write(str(end_time - start_time) + "\n")
            
        print(f"Saved {num_written} outputs to {file_save_path}")
        print(f"Total time taken: {end_time - start_time}")

    else:
        
        num_written = 0
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        with open(file_save_path, 'w') as f:
            for idx in range(args.num_test_prompts):
                row = dataset.iloc[idx]
                
                if is_bbm:
                    prompt = row['input']
                    generation = format_bbm_steps([row['steps']])[0]
                elif args.dataset == "toxicspans_extended":
                    prompt = ""
                    generation = row['text']
                else:
                    prompt = row['prompt']
                    generation = row['generation']

                if is_bbm:
                    user_content = template.format(input=prompt, steps=generation)
                    messages = messages_base + [{"role": "user", "content": user_content}]
                else:
                    user_content = (user_prompt % (prompt, generation)) if prompt != "" else (user_prompt % generation)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                if "Qwen3-8B" in args.model:
                    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                else:
                    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # encode prompt
                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                
                if args.model.split('/')[-1] == 'Qwen3-8B':
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            min_p=0,
                            max_new_tokens=args.max_tokens,
                            num_return_sequences=args.num_return_sequences,
                        )
                        
                    generated_text_raw = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=False)
                    
                    if '</think>' in generated_text_raw:
                        content = generated_text_raw.split('</think>')[-1].strip()
                    elif r"<\/think>" in generated_text_raw:
                        content = generated_text_raw.split(r"<\/think>")[-1].strip()
                    else:
                        content = None # Truncated/Invalid

                    # 2. Format based on task and success
                    if content is not None:
                        if is_bbm:
                            try:
                                parsed = json.loads(content)
                                if isinstance(parsed, dict):
                                    generated_text = parsed
                                else:
                                    generated_text = {"answer": str(parsed)}
                            except:
                                generated_text = {"answer": content}
                        else:
                            try:
                                generated_text = json.loads(content)
                            except:
                                generated_text = {"spans": [], "error": "failed to parse json"}
                    else:
                        # 3. Handle truncated responses
                        if is_bbm:
                            generated_text = {"answer": "error: truncated", "reasoning": "truncated before </think>"}
                        else:
                            generated_text = {"spans": [], "error": "truncated before </think>"}
                    f.write(json.dumps(generated_text) + '\n')
                    num_written += 1
                
                elif args.model.split('/')[-1] == 'QwQ-32B-Preview':

                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            do_sample=args.temperature > 0,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_tokens,
                            num_return_sequences=args.num_return_sequences,
                        )
                        
                    generated_text_raw = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=False)
 
                    f.write(generated_text_raw + '\n')
                    num_written += 1

                else:
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            do_sample=args.temperature > 0,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_tokens,
                            num_return_sequences=args.num_return_sequences,
                        )
                        
                    generated_text_raw = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=False)
    
                    f.write(json.dumps(generated_text_raw) + '\n')
                    num_written += 1

        end_time = time.time()
        with open(execution_time_path, 'w') as g:
            g.write(str(end_time - start_time) + "\n")
        print(f"Saved {num_written} outputs to {file_save_path}")
        print(f"Total time taken: {end_time - start_time}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('prompt_type', type=str)
    parser.add_argument('dataset', type=str, choices=["toxicspans", "toxicspans_extended", "inconsistentspans", "logical_deduction", "tracking_shuffled_objects"])
    parser.add_argument('--num_test_prompts', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=30)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--reasoning_effort', type=str, default="medium")
    parser.add_argument('--use_vllm', action='store_true')
    
    args = parser.parse_args()

    # load data
    print(os.getcwd())
    
    if args.dataset == "toxicspans":
        dataset = pd.read_json("laser_edit/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl", lines=True)
    
    elif args.dataset == "toxicspans_extended":
        dataset = pd.read_json("laser_edit/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl", lines=True)
    
    elif args.dataset == "inconsistentspans":
        dataset = pd.read_json("laser_edit/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl", lines=True)
        dataset = dataset.rename(columns = {"premise": "prompt",
                                  "hypothesis": "generation"},)
    
    elif args.dataset == "logical_deduction":
        dataset = pd.read_json("laser_edit/data/BIG-Bench-Mistake/logical_deduction.jsonl", lines=True)
        
    elif args.dataset == "tracking_shuffled_objects":
        dataset = pd.read_json("laser_edit/data/BIG-Bench-Mistake/tracking_shuffled_objects.jsonl", lines=True)
    

    # run locate 
    
    if ("gpt" in args.model.lower()) or ("o3" in args.model.lower()) or ("o4" in args.model.lower()) or ("o1" in args.model.lower()): 
        run_api_and_save_result(dataset, args)
    else:
        run_whitebox_and_save_result(dataset, args)
