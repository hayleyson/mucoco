import os
import json
import argparse
import time
import importlib.util
import sys

from tqdm import tqdm
from openai import OpenAI
import pandas as pd

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    LLM = None
    SamplingParams = None
    AutoTokenizer = None

from new_module.llm_experiments.locate_with_llm.prompts import get_prompt

def get_bbm_prompts(dataset_name, prompt_type):
    """Load BBM prompts from external files."""
    prompt_file = f"new_module/data/BIG-Bench-Mistake/mistake_finding_prompts/{dataset_name}_prompts.py"
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
    file_save_path = f"new_module/llm_experiments/locate_with_llm/results/{args.model}_{args.prompt_type}_{args.dataset}_{str(int(start_time))}.jsonl"
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
        for p, g in tqdm(zip(prompts[256:max_prompt_count], generations[256:max_prompt_count])):
            
            response = client.responses.create(
                model = args.model,
                reasoning = {"effort": "medium"},
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
    file_save_path = f"new_module/llm_experiments/locate_with_llm/results/{args.model.split('/')[-1]}_{args.prompt_type}_{args.dataset}_{str(int(start_time))}.jsonl"
    execution_time_path = file_save_path.replace(".jsonl", ".time")

    is_bbm = args.dataset in ["logical_deduction", "tracking_shuffled_objects"]
    
    if is_bbm:
        messages_base, template = get_bbm_prompts(args.dataset, args.prompt_type)
    else:
        system_prompt, user_prompt = get_prompt(args)
    
    print(f"Initializing vLLM with {args.model}...")
    llm = LLM(model=args.model, gpu_memory_utilization=0.8)
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
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_for_vllm.append(prompt_text)

    print("Running vLLM inference...")
    outputs = llm.generate(inputs_for_vllm, sampling_params)
    
    with open(file_save_path, 'w') as f:
        for output in outputs:
            generated_text_raw = output.outputs[0].text
            
            if args.model.split('/')[-1] == 'Qwen3-8B':
                # 1. Extract content after thinking process
                if '</think>' in generated_text_raw:
                    content = generated_text_raw.split('</think>')[-1].strip()
                elif "<\/think>" in generated_text_raw:
                    content = generated_text_raw.split("<\/think>")[-1].strip()
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

            elif args.model.split('/')[-1] == 'QwQ-32B-Preview':

                f.write(generated_text_raw + '\n')
            
    end_time = time.time()
    with open(execution_time_path, 'w') as f:
        f.write(str(end_time - start_time) + "\n")
        
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
    
    elif args.dataset == "logical_deduction":
        dataset = pd.read_json("new_module/data/BIG-Bench-Mistake/logical_deduction.jsonl", lines=True)
        
    elif args.dataset == "tracking_shuffled_objects":
        dataset = pd.read_json("new_module/data/BIG-Bench-Mistake/tracking_shuffled_objects.jsonl", lines=True)
    

    # run locate 
    
    if ("gpt" in args.model.lower()) or ("o3" in args.model.lower()) or ("o4" in args.model.lower()) or ("o1" in args.model.lower()): 
        run_api_and_save_result(dataset, args)
    else:
        run_whitebox_and_save_result(dataset, args)
