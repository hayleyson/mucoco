# copy of llm_generate
# toxicity jsonl 파일 처리 위함
# sentiment jsonl 파일도 (dev_set_36184.jsonl)

import joblib
import json
import argparse
import os
import time
import re
import sys


import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import huggingface_hub
import pandas as pd

from new_module.llm_experiments.edit_with_llm.prompts import get_prompt
from new_module.set_consistency_energy.baselines.LLM.lm_loader import lm_loader
from new_module.dev_utils.utils import parse_set_text, convert_format

from openai import OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HF_TOKEN")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


huggingface_hub.login(token=huggingface_token)

def postprocess(model, text):
    new_text = ''
    if 'llama' in model.lower():
        new_text = text.strip('\n')
    elif 'phi' in model.lower():
        new_text = text.split('\n')[0]
        new_text = new_text.split('.')[0]
    elif 'mistral' in model.lower():
        new_text = text.split('.')[0]
    elif 'gemma' in model.lower():
        new_text = text.split('\n')[0]
    else:
      new_text = text

    return new_text

def generate_and_save_result(args):
    
    # run = wandb.init(project="llm_experiments", entity="saehee-seoul-national-university", config=vars(args))
    device= "cuda" if torch.cuda.is_available() else "cpu"

    # Load model directly
    # Suppose you conducted huggingface-cli login and authenticated with your auth token
    print('Loading tokenizer...')
    #hf_model_name = "/home/saeheeeom/data/set_consistency/llama3.1_8b"
    hf_model_name = args.hf_model_name
    if 'mistral' in args.hf_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            revision="f67d0f47df7707eddf3fb61000e3e8713074f45c"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    
    print("Done!")

    print(f'EOS token: {tokenizer.eos_token}')
    print(f'PAD token: {tokenizer.pad_token}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    if '70b' in args.hf_model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_name, device_map="auto")
    elif 'mistral' in args.hf_model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            revision="f67d0f47df7707eddf3fb61000e3e8713074f45c",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else: 
        model = AutoModelForCausalLM.from_pretrained(hf_model_name)
        
        #model = model.half()
    model = model.to(device)
    print("Done!")

    print("Loading data...")

    # for 'both'
    # orig_text_path = ""
    orig_prompts = []
    orig_text_lists = []
    orig_text_path = args.orig_text_path
    # if '0shot' in args.input_file_path:
    #     orig_text_path = "/home/saeheeeom/data/set_consistency/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl"
    # else:
    #     orig_text_path = '/data3/saeheeeom/set_consistency/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_noprompt_150.jsonl'

    # for sentiment dev_set.jsonl
    #orig_text_path = "/data3/saeheeeom/set_consistency/data/sentiment/dev_set.jsonl"
    print("orig_text_path:", orig_text_path)
    
    locate_edit_idx = args.locate_edit_idx

    if len(locate_edit_idx) == 0:
        with open(orig_text_path,'r') as f:
            raw_data = f.readlines()
            for line in raw_data:
                tmp_list = []
                line = json.loads(line)
                p = line['prompt']['text']
                orig_prompts.append(p)
                gens = line['generations']
                for gen in gens:
                    tmp_list.append(gen['text'])
                orig_text_lists.append(tmp_list)
    else:
        with open(orig_text_path,'r') as f:
            raw_data = f.readlines()
            for line_idx, line in enumerate(raw_data):
                tmp_list = []
                line = json.loads(line)
                p = line['prompt']['text']
                orig_prompts.append(p)
                gens = line['generations']
                for gen_idx, gen in enumerate(gens):
                    if locate_edit_idx[line_idx][gen_idx]:
                        tmp_list.append(gen['text'])
                if tmp_list:
                    orig_text_lists.append(tmp_list)

    enable_thinking = args.enable_thinking

    # these 'prompt's are the strings that are formatted into the prompt
    prompts = []
    num_generations_per_prompt = []
    with open(args.input_file_path,'r') as f:
        raw_data = f.readlines()
        for lineidx, line in enumerate(raw_data):
            line = json.loads(line)
            prompt = line['prompt']['text']
            gens = line['generations']
            num_generations_per_prompt.append(len(gens))
            # 인덱스 검증 및 디버깅 정보
            if lineidx >= len(orig_text_lists):
                print(f"Error: lineidx {lineidx} out of range for orig_text_lists (len={len(orig_text_lists)})")
                continue

            if len(orig_text_lists[lineidx]) != len(gens):
                print(f"Error: Mismatch in generations at lineidx {lineidx}: "
                    f"orig_text_lists has {len(orig_text_lists[lineidx])}, gens has {len(gens)}")
                continue
            
            for genidx, gen in enumerate(gens):
                gen = gen['text']
                if 'form' in args.prompt_type:
                    if 'notmasked' in args.prompt_type:
                        concatenated = orig_text_lists[lineidx][genidx]
                    elif 'masked'in args.prompt_type:
                        concatenated = gen
                    else: # both
                        concatenated = orig_text_lists[lineidx][genidx] + '\nMasked Sequence: ' + gen
                elif 'nli' in args.prompt_type:
                    if 'notmasked' in args.prompt_type:
                        concatenated = prompt + '\nHypothesis: ' + orig_text_lists[lineidx][genidx]
                    elif 'masked'in args.prompt_type:
                        concatenated = prompt + '\nMasked Hypothesis: ' + gen
                    else: # both
                        concatenated = prompt + '\nOriginal Hypothesis: ' + orig_text_lists[lineidx][genidx] + '\nMasked Hypothesis: ' + gen
                else:
                    if 'notmasked' in args.prompt_type:
                        concatenated = prompt + '\nText: ' + orig_text_lists[lineidx][genidx]
                    elif 'masked'in args.prompt_type:
                        concatenated = prompt + '\nMasked Text: ' + gen
                    else: # both
                        concatenated = prompt + '\nOriginal Text: ' + orig_text_lists[lineidx][genidx] + '\nMasked Text: ' + gen
                prompts.append(concatenated)

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

    def apply_ph(text, ph):
        new_text = re.sub(r'(<mask>\s*)+', '<mask>', text)
        new_text = new_text.replace('<mask>', ph)
        return new_text
    
    # setting for llama, mistral, phi
    if 'llama' in args.hf_model_name.lower():
        nontoxic_prompt = re.sub(r'(<mask>\s*)+', '<mask>', nontoxic_prompt)
        nontoxic_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + nontoxic_prompt
        nontoxic_prompt = nontoxic_prompt.replace("\nGiven", "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nGiven")
        nontoxic_prompt = nontoxic_prompt + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        prompts = [apply_ph(p, '<mask>') for p in prompts]

    elif 'phi' in hf_model_name.lower():
        nontoxic_prompt = "<|system|>" + nontoxic_prompt
        nontoxic_prompt = nontoxic_prompt.replace("\nGiven", "<|end|>\n<|user|>\nGiven")
        nontoxic_prompt = nontoxic_prompt + "\n<|end|>\n<|assistant|>"

    elif 'mistral' in args.hf_model_name.lower():
        nontoxic_prompt = "<s>[INST] <<SYS>>" + nontoxic_prompt + "[/INST]"
        prompt_tmp = nontoxic_prompt.split("Given")
        nontoxic_prompt = "<</SYS>>\nGiven".join(prompt_tmp)
        nontoxic_prompt = re.sub(r'(<mask>\s*)+', '<mask>', nontoxic_prompt)
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '[BLANK]')
        prompts = [apply_ph(p, '[BLANK]') for p in prompts]
    
    elif 'qwen2.5' in args.hf_model_name.lower():
        sp = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. All your responses must be in English.<|im_end|>
<|im_start|>user\n"""
        nontoxic_prompt = sp + nontoxic_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        prompts = [apply_ph(p, "___") for p in prompts]
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '___')
        
    elif 'qwen3' in args.hf_model_name.lower():
        prompts = [apply_ph(p, "___") for p in prompts]
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '___')
        messages = [
            {"role": "user", "content": nontoxic_prompt}
        ]
        nontoxic_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
        )
        
    elif 'gemma' in args.hf_model_name.lower():
        nontoxic_prompt_splits = nontoxic_prompt.split("\nEdited Text: ")
        sp = "<bos><start_of_turn>user\n"
        nontoxic_prompt_0 = sp + nontoxic_prompt_splits[0] + "<end_of_turn>\n"
        nontoxic_prompt_1 = "<start_of_turn>model\n"
        nontoxic_prompt = nontoxic_prompt_0 + nontoxic_prompt_1
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '___')
        prompts = [apply_ph(p, "___") for p in prompts]

    print("nontoxic_prompt:\n", nontoxic_prompt)

    # set decoding params
    if 'qwen3' in args.hf_model_name.lower() and enable_thinking:
        decoding_params = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0
        }
    elif 'qwen3' in args.hf_model_name.lower() and not enable_thinking:
        decoding_params = {
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.7,
            "top_k": 20,
            "min_p": 0
        }
    else:
        decoding_params = {
            "do_sample": True,
            "top_p": 0.96,
            "temperature": 1.0,
        }
    
    # start generation
    print("=========== start generation! =============")

    f = open(args.file_save_path, 'w')

    myDataset = CustomDataset(prompts, nontoxic_prompt)
    myCollateFn = CollateFnClass(tokenizer)
    myDataLoader = DataLoader(myDataset, batch_size=1, collate_fn=myCollateFn.collate_fn)
    print("Done!")
    
    start_time = time.time()
    count = 0
    edited_text   = []
    prompt_idx = 0
    processed_generations = 0
    for prompt, (batch, batch_text) in zip(prompts, myDataLoader):
        batch = batch.to(device)
        generated_result = model.generate(**batch, 
                                        max_length=batch.input_ids.shape[-1] + args.max_tokens,
                                        num_return_sequences=args.num_return_sequences,
                                        **decoding_params)
        
        input_length = batch.input_ids.shape[-1]
        
        generated_tokens = generated_result[:, input_length:]
        # print("raw result: \n", tokenizer.convert_ids_to_tokens(generated_result[0]))
        if 'qwen3' in args.hf_model_name.lower():
            try:
                print(f"generated_tokens: {generated_tokens}")
                index = [len(seq) - seq[::-1].index(151668) for seq in generated_tokens]
                print(f"index: {index}")
            except Exception as e:
                print(f"Error: {e}")
                index = [0 for _ in range(args.num_return_sequences)]
            thinking_content = [tokenizer.decode(seq[:ix], skip_special_tokens=True).strip('\n') for seq, ix in zip(generated_tokens, index)]
            total_generated_text = [tokenizer.decode(seq[ix:], skip_special_tokens=True).strip('\n') for seq, ix in zip(generated_tokens, index)]
        else:
            total_generated_text = [tokenizer.decode(seq, skip_special_tokens=True).strip('\n') for seq in generated_tokens]
        edited_text.append(total_generated_text[0])
        if len(edited_text) == num_generations_per_prompt[prompt_idx]:
            formatted_generated_text = {
                'prompt': {'text': prompts[processed_generations].split("\n")[0].replace("Prompt: ", "")},
                'generations': [{'text': x.replace("\n\n", "")} for x in edited_text]
            }
            f.write(json.dumps(formatted_generated_text, ensure_ascii=False) + '\n')
            f.flush()
            processed_generations += len(edited_text)
            edited_text = []
            prompt_idx += 1
        count += 1
        #if count >= 10:
            #break
        if count == 1:
            print("example prompt\n============\n" + nontoxic_prompt + "\n============\n" + prompt)

    f.close()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")


def generate_and_save_result_vllm(args):
    from vllm import LLM, SamplingParams

    print("Loading tokenizer...")
    hf_model_name = args.hf_model_name
    if 'mistral' in args.hf_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            revision="f67d0f47df7707eddf3fb61000e3e8713074f45c"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    
    print("Done!")

    print(f'EOS token: {tokenizer.eos_token}')
    print(f'PAD token: {tokenizer.pad_token}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")

    # for 'both'
    orig_prompts = []
    orig_text_lists = []
    orig_text_path = args.orig_text_path
    print("orig_text_path:", orig_text_path)

    locate_edit_idx = args.locate_edit_idx

    if len(locate_edit_idx) == 0:
        with open(orig_text_path,'r') as f:
            raw_data = f.readlines()
            for line in raw_data:
                tmp_list = []
                line = json.loads(line)
                p = line['prompt']['text']
                orig_prompts.append(p)
                gens = line['generations']
                for gen in gens:
                    tmp_list.append(gen['text'])
                orig_text_lists.append(tmp_list)
    else:
        with open(orig_text_path, "r") as f:
            raw_data = f.readlines()
            for line_idx, line in enumerate(raw_data):
                tmp_list = []
                line = json.loads(line)
                p = line['prompt']['text']
                orig_prompts.append(p)
                gens = line["generations"]
                for gen_idx, gen in enumerate(gens):
                    if locate_edit_idx[line_idx][gen_idx]:
                        tmp_list.append(gen['text'])
                if tmp_list:
                    orig_text_lists.append(tmp_list)

    enable_thinking = args.enable_thinking

    prompts = []
    num_generations_per_prompt = []
    with open(args.input_file_path,'r') as f:
        raw_data = f.readlines()
        for lineidx, line in enumerate(raw_data):
            line = json.loads(line)
            prompt = line['prompt']['text']
            gens = line['generations']
            num_generations_per_prompt.append(len(gens))
            if lineidx >= len(orig_text_lists):
                print(f"Error: lineidx {lineidx} out of range for orig_text_lists (len={len(orig_text_lists)})")
                continue

            if len(orig_text_lists[lineidx]) != len(gens):
                print(f"Error: Mismatch in generations at lineidx {lineidx}: "
                    f"orig_text_lists has {len(orig_text_lists[lineidx])}, gens has {len(gens)}")
                continue
            
            for genidx, gen in enumerate(gens):
                gen = gen['text']
                if 'form' in args.prompt_type:
                    if 'notmasked' in args.prompt_type:
                        concatenated = orig_text_lists[lineidx][genidx]
                    elif 'masked'in args.prompt_type:
                        concatenated = gen
                    else: # both
                        concatenated = orig_text_lists[lineidx][genidx] + '\nMasked Sequence: ' + gen
                elif 'nli' in args.prompt_type:
                    if 'notmasked' in args.prompt_type:
                        concatenated = prompt + '\nHypothesis: ' + orig_text_lists[lineidx][genidx]
                    elif 'masked'in args.prompt_type:
                        concatenated = prompt + '\nMasked Hypothesis: ' + gen
                    else: # both
                        concatenated = prompt + '\nOriginal Hypothesis: ' + orig_text_lists[lineidx][genidx] + '\nMasked Hypothesis: ' + gen
                else:
                    if 'notmasked' in args.prompt_type:
                        concatenated = prompt + '\nText: ' + orig_text_lists[lineidx][genidx]
                    elif 'masked'in args.prompt_type:
                        concatenated = prompt + '\nMasked Text: ' + gen
                    else: # both
                        concatenated = prompt + '\nOriginal Text: ' + orig_text_lists[lineidx][genidx] + '\nMasked Text: ' + gen
                prompts.append(concatenated)

    nontoxic_prompt = get_prompt(args)

    def apply_ph(text, ph):
        new_text = re.sub(r'(<mask>\s*)+', '<mask>', text)
        new_text = new_text.replace('<mask>', ph)
        return new_text
    
    # setting for llama, mistral, phi
    if 'llama' in args.hf_model_name.lower():
        nontoxic_prompt = re.sub(r'(<mask>\s*)+', '<mask>', nontoxic_prompt)
        nontoxic_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + nontoxic_prompt
        nontoxic_prompt = nontoxic_prompt.replace("\nGiven", "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nGiven")
        nontoxic_prompt = nontoxic_prompt + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        prompts = [apply_ph(p, '<mask>') for p in prompts]

    elif 'phi' in hf_model_name.lower():
        nontoxic_prompt = "<|system|>" + nontoxic_prompt
        nontoxic_prompt = nontoxic_prompt.replace("\nGiven", "<|end|>\n<|user|>\nGiven")
        nontoxic_prompt = nontoxic_prompt + "\n<|end|>\n<|assistant|>"

    elif 'mistral' in args.hf_model_name.lower():
        nontoxic_prompt = "<s>[INST] <<SYS>>" + nontoxic_prompt + "[/INST]"
        prompt_tmp = nontoxic_prompt.split("Given")
        nontoxic_prompt = "<</SYS>>\nGiven".join(prompt_tmp)
        nontoxic_prompt = re.sub(r'(<mask>\s*)+', '<mask>', nontoxic_prompt)
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '[BLANK]')
        prompts = [apply_ph(p, '[BLANK]') for p in prompts]
    
    elif 'qwen2.5' in args.hf_model_name.lower():
        sp = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. All your responses must be in English.<|im_end|>
<|im_start|>user\n"""
        nontoxic_prompt = sp + nontoxic_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        prompts = [apply_ph(p, "___") for p in prompts]
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '___')
        
    elif 'qwen3' in args.hf_model_name.lower():
        prompts = [apply_ph(p, "___") for p in prompts]
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '___')
        messages = [
            {"role": "user", "content": nontoxic_prompt}
        ]
        nontoxic_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
        )
        
    elif 'gemma' in args.hf_model_name.lower():
        nontoxic_prompt_splits = nontoxic_prompt.split("\nEdited Text: ")
        sp = "<bos><start_of_turn>user\n"
        nontoxic_prompt_0 = sp + nontoxic_prompt_splits[0] + "<end_of_turn>\n"
        nontoxic_prompt_1 = "<start_of_turn>model\n"
        nontoxic_prompt = nontoxic_prompt_0 + nontoxic_prompt_1
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '___')
        prompts = [apply_ph(p, "___") for p in prompts]

    print("nontoxic_prompt:\n", nontoxic_prompt)

    full_prompts = [nontoxic_prompt % p for p in prompts]

    if "qwen3" in args.hf_model_name.lower() and enable_thinking:
        decoding_params = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0
        }
    elif 'qwen3' in args.hf_model_name.lower() and not enable_thinking:
        decoding_params = {
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.7,
            "top_k": 20,
            "min_p": 0
        }
    else:
        decoding_params = {"do_sample": True, "top_p": 0.96, "temperature": 1.0}

    sp_kwargs = {
        "max_tokens": args.max_tokens,
        "n": args.num_return_sequences,
    }
    if decoding_params.get("do_sample", True):
        sp_kwargs["temperature"] = decoding_params["temperature"]
        sp_kwargs["top_p"] = decoding_params["top_p"]
        if "top_k" in decoding_params:
            sp_kwargs["top_k"] = decoding_params["top_k"]
        if "min_p" in decoding_params:
            sp_kwargs["min_p"] = decoding_params["min_p"]
    else:
        sp_kwargs["temperature"] = 0.0

    sampling_params = SamplingParams(**sp_kwargs)

    vllm_model = args.hf_model_name
    vllm_revision = None
    if "mistral" in args.hf_model_name.lower():
        vllm_model = "mistralai/Mistral-7B-Instruct-v0.3"
        vllm_revision = "f67d0f47df7707eddf3fb61000e3e8713074f45c"

    tp = getattr(args, "vllm_tensor_parallel_size", 1)
    gpu_mem = getattr(args, "vllm_gpu_memory_utilization", 0.9)
    max_model_len = getattr(args, "vllm_max_model_len", None)
    chunk_size = getattr(args, "vllm_prompt_chunk_size", None)
    trust_remote_code = getattr(args, "vllm_trust_remote_code", True)

    llm_kw = dict(
        model=vllm_model,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        dtype=getattr(args, "vllm_dtype", "auto"),
    )
    if vllm_revision is not None:
        llm_kw["revision"] = vllm_revision
    if max_model_len is not None:
        llm_kw["max_model_len"] = max_model_len

    print("Loading vLLM engine...")
    llm = LLM(**llm_kw)
    print("Done!")

    print("=========== start generation (vLLM)! =============")

    f = open(args.file_save_path, "w")

    start_time = time.time()
    all_outputs = []
    if chunk_size is None or chunk_size <= 0:
        all_outputs = llm.generate(full_prompts, sampling_params)
    else:
        for start in range(0, len(full_prompts), chunk_size):
            all_outputs.extend(
                llm.generate(full_prompts[start : start + chunk_size], sampling_params)
            )

    def _first_completion_text(req_out):
        out0 = req_out.outputs[0]
        if "qwen3" not in args.hf_model_name.lower():
            return out0.text.strip("\n").replace("\n\n", "")
        gen_ids = list(out0.token_ids)
        try:
            ix = len(gen_ids) - gen_ids[::-1].index(151668)
        except ValueError:
            ix = 0
        tail = tokenizer.decode(gen_ids[ix:], skip_special_tokens=True).strip("\n")
        return tail.replace("\n\n", "")

    edited_text = []
    prompt_idx = 0
    processed_generations = 0
    for count, req_out in enumerate(all_outputs):
        prompt = prompts[count]
        if count == 0:
            print(
                "example prompt\n============\n"
                + nontoxic_prompt
                + "\n============\n"
                + prompt
            )
        text = _first_completion_text(req_out)
        edited_text.append(text)
        if len(edited_text) == num_generations_per_prompt[prompt_idx]:
            formatted_generated_text = {
                'prompt': {'text': prompts[processed_generations].split("\n")[0].replace("Prompt: ", "")},
                'generations': [{'text': x.replace("\n\n", "")} for x in edited_text]
            }
            f.write(json.dumps(formatted_generated_text, ensure_ascii=False) + "\n")
            f.flush()
            processed_generations += len(edited_text)
            edited_text = []
            prompt_idx += 1

    f.close()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")


def generate_and_save_result_vllm_sc_energy(args, test_dataset, located_dataset):
    from vllm import LLM, SamplingParams

    print("Loading tokenizer...")
    hf_model_name = args.hf_model_name
    if 'mistral' in args.hf_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            revision="f67d0f47df7707eddf3fb61000e3e8713074f45c"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    
    print("Done!")

    print(f'EOS token: {tokenizer.eos_token}')
    print(f'PAD token: {tokenizer.pad_token}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")

    # for 'both'
    orig_prompts = []
    orig_text_lists = []
    orig_text_path = args.orig_text_path
    print("orig_text_path:", orig_text_path)

    locate_edit_idx = args.locate_edit_idx


    orig_data_loader = [lm_loader(dataset=test_dataset, tokenizer= tokenizer, params=vars(args)).get_loader()]
    located_data_loader = [lm_loader(dataset=located_dataset, tokenizer= tokenizer, params=vars(args)).get_loader()]

    enable_thinking = args.enable_thinking

    line_idx = 0
    prompts = []
    for batch, batch_orig in zip(located_data_loader, orig_data_loader):
        if not locate_edit_idx[line_idx][0]:
            line_idx += 1
            continue
        for _idx, line in enumerate(batch):
            prompt = ""
            gen = line[0]
            orig_gen = batch_orig[_idx][0]
        
            if 'notmasked' in args.prompt_type:
                concatenated = prompt + '\nText: ' + orig_gen
            elif 'masked'in args.prompt_type:
                concatenated = prompt + '\nMasked Text: ' + gen
            else: # both
                concatenated = prompt + '\nOriginal Text: ' + orig_gen + '\nMasked Text: ' + gen
            prompts.append(concatenated)

    nontoxic_prompt = get_prompt(args)

    def apply_ph(text, ph):
        new_text = re.sub(r'(<mask>\s*)+', '<mask>', text)
        new_text = new_text.replace('<mask>', ph)
        return new_text
    
    # setting for llama, mistral, phi
    if 'llama' in args.hf_model_name.lower():
        prompts = [apply_ph(p, '<mask>') for p in prompts]
    
    elif 'mistral' in args.hf_model_name.lower():
        prompts = [apply_ph(p, '[BLANK]') for p in prompts]
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '[BLANK]')
    
    elif ('qwen' in args.hf_model_name.lower()) or ('gemma' in args.hf_model_name.lower()):
        prompts = [apply_ph(p, "___") for p in prompts]
        nontoxic_prompt = nontoxic_prompt.replace('<mask>', '___')
    
    nontoxic_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": nontoxic_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
    )
     
    print("nontoxic_prompt:\n", nontoxic_prompt)

    full_prompts = [nontoxic_prompt % p for p in prompts]

    if "qwen3" in args.hf_model_name.lower() and enable_thinking:
        decoding_params = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0
        }
    elif 'qwen3' in args.hf_model_name.lower() and not enable_thinking:
        decoding_params = {
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.7,
            "top_k": 20,
            "min_p": 0
        }
    else:
        decoding_params = {"do_sample": True, "top_p": 0.96, "temperature": 1.0}

    sp_kwargs = {
        "max_tokens": args.max_tokens,
        "n": args.num_return_sequences,
    }
    if decoding_params.get("do_sample", True):
        sp_kwargs["temperature"] = decoding_params["temperature"]
        sp_kwargs["top_p"] = decoding_params["top_p"]
        if "top_k" in decoding_params:
            sp_kwargs["top_k"] = decoding_params["top_k"]
        if "min_p" in decoding_params:
            sp_kwargs["min_p"] = decoding_params["min_p"]
    else:
        sp_kwargs["temperature"] = 0.0

    sampling_params = SamplingParams(**sp_kwargs)

    vllm_model = args.hf_model_name
    vllm_revision = None
    if "mistral" in args.hf_model_name.lower():
        vllm_model = "mistralai/Mistral-7B-Instruct-v0.3"
        vllm_revision = "f67d0f47df7707eddf3fb61000e3e8713074f45c"

    tp = getattr(args, "vllm_tensor_parallel_size", 1)
    gpu_mem = getattr(args, "vllm_gpu_memory_utilization", 0.9)
    max_model_len = getattr(args, "vllm_max_model_len", None)
    chunk_size = getattr(args, "vllm_prompt_chunk_size", None)
    trust_remote_code = getattr(args, "vllm_trust_remote_code", True)

    llm_kw = dict(
        model=vllm_model,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        dtype=getattr(args, "vllm_dtype", "auto"),
    )
    if vllm_revision is not None:
        llm_kw["revision"] = vllm_revision
    if max_model_len is not None:
        llm_kw["max_model_len"] = max_model_len

    print("Loading vLLM engine...")
    llm = LLM(**llm_kw)
    print("Done!")

    print("=========== start generation (vLLM)! =============")

    f = open(args.file_save_path, "w")

    start_time = time.time()
    all_outputs = []
    if chunk_size is None or chunk_size <= 0:
        all_outputs = llm.generate(full_prompts, sampling_params)
    else:
        for start in range(0, len(full_prompts), chunk_size):
            all_outputs.extend(
                llm.generate(full_prompts[start : start + chunk_size], sampling_params)
            )

    def _first_completion_text(req_out):
        out0 = req_out.outputs[0]
        if "qwen3" not in args.hf_model_name.lower():
            return out0.text.strip("\n").replace("\n\n", "")
        gen_ids = list(out0.token_ids)
        try:
            ix = len(gen_ids) - gen_ids[::-1].index(151668)
        except ValueError:
            ix = 0
        tail = tokenizer.decode(gen_ids[ix:], skip_special_tokens=True).strip("\n")
        return tail.replace("\n\n", "")

    edited_text = []
    edited_set_texts = []
    for count, req_out in enumerate(all_outputs):
        prompt = ""
        if count == 0:
            print(
                "example prompt\n============\n"
                + nontoxic_prompt
                + "\n============\n"
                + prompt
            )
        text = _first_completion_text(req_out)
        edited_text.append(text)
        edited_ebm_text = convert_format(text, source_mode="llm", target_mode="ebm")
        
        formatted_generated_text = {
                'prompt': {'text': ""},
                'generations': [{'text': edited_ebm_text}]
            }
        f.write(json.dumps(formatted_generated_text, ensure_ascii=False) + "\n")
        f.flush()
        
        edited_set_text = parse_set_text(text, source_mode="llm")
        edited_set_texts.append(edited_set_text)

    f.close()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    return edited_set_texts

    
def generate_and_save_result_gpt(args):
    
    # run = wandb.init(project="llm_experiments", entity="saehee-seoul-national-university", config=vars(args))
    device= "cuda" if torch.cuda.is_available() else "cpu"

    
    print("Loading model...")
    # TODO: load gpt 3.5 turbo 0125
    print("Calling openAI API...")
    client = OpenAI(api_key=openai_api_key)
    print("Calling done!\n")

    print("Loading data...")

    # for 'both'
    orig_text_path = ""
    orig_prompts = []
    orig_text_lists = []
    """
    if '0shot' in args.input_file_path:
        orig_text_path = "/home/saeheeeom/data/set_consistency/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl"
    else:
        orig_text_path = '/data3/saeheeeom/set_consistency/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_noprompt_150.jsonl'
    with open(orig_text_path,'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            tmp_list = []
            line = json.loads(line)
            p = line['prompt']['text']
            orig_prompts.append(p)
            gens = line['generations']
            for gen in gens:
                tmp_list.append(gen['text'])
            orig_text_lists.append(tmp_list)
"""
    
    with open(orig_text_path,'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            tmp_list = []
            line = json.loads(line)
            p = line['prompt']['text']
            orig_prompts.append(p)
            gens = line['generations']
            for gen in gens:
                tmp_list.append(gen['text'])
            orig_text_lists.append(tmp_list)

    prompts = []
    with open(args.input_file_path,'r') as f:
        raw_data = f.readlines()
        for lineidx, line in enumerate(raw_data):
            line = json.loads(line)
            prompt = line['prompt']['text']
            gens = line['generations']
            for genidx, gen in enumerate(gens):
                gen = gen['text']
                if 'notmasked' in args.prompt_type:
                    concatenated = "Prompt: " + prompt + '\nText: ' + orig_text_lists[lineidx][genidx]
                elif 'masked'in args.prompt_type:
                    concatenated = "Prompt: " + prompt + '\nMasked Text: ' + gen
                else: # both
                    concatenated = "Prompt: " + prompt + '\nOriginal Text: ' + orig_text_lists[lineidx][genidx] + '\nMasked Text: ' + gen
                concatenated = concatenated + "\nEdited Text: "
                prompts.append(concatenated)


    nontoxic_prompt = get_prompt(args)
    nontoxic_prompt = nontoxic_prompt.split("Prompt: ")[0]

    def apply_ph(text, ph):
        new_text = re.sub(r'(<mask>\s*)+', '<mask>', text)
        new_text = new_text.replace('<mask>', ph)
        return new_text
    
    print("nontoxic_prompt:\n", nontoxic_prompt)

    
    # start generation
    print("=========== start generation! =============")

    f = open(args.file_save_path, 'w')

    start_time = time.time()
    count = 0
    edited_text = []
    for prompt in prompts:
        messages = [{"role": "system", "content": nontoxic_prompt},
                    {"role": "user", "content":prompt}]
        
        response = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = messages
    )
        bot_text = response.choices[0].message.content
        edited_text.append(bot_text)

        # how many 'text's are there in the 'generations' list?
        # ex: 20 for dev_set.jsonl (sentiment), 10 for the rest of the datasets
        txt_per_gen = 20
        prompt_idx = count // txt_per_gen
        gen_idx = count % txt_per_gen

        if gen_idx == (txt_per_gen - 1):
            formatted_generated_text = {'prompt': {'text': prompt.split("\n")[0].replace("Prompt: ", "")},
                                        'generations': [{'text': x} for x in edited_text]}
            f.write(json.dumps(formatted_generated_text, ensure_ascii=False) + '\n')
            f.flush()
            edited_text = []
        
        count += 1
        #if count >= 10:
            #break
        if count == 1:
            print("example prompt\n============\n" + nontoxic_prompt + "\n============\n" + prompt)
    f.close()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    #run.summary["execution_time"] = (end_time - start_time)
    #run.finish()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str)
    parser.add_argument('--file_save_path', type=str)
    parser.add_argument('--input_file_path', type=str)
    parser.add_argument('--orig_text_path', type=str)
    parser.add_argument('--locate_edit_idx', type=list, default=[])
    parser.add_argument('--prompt_type', type=str)
    parser.add_argument('--num_return_sequences', type=int, default=10)
    parser.add_argument('--max_tokens', type=int, default=30)
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--run_type', type=str)
    parser.add_argument('--use_vllm', action='store_true', help='Use vLLM in generate_and_save_result_vllm (faster on GPU)')

    args = parser.parse_args()
    _gen_hf_or_vllm = generate_and_save_result_vllm if args.use_vllm else generate_and_save_result
    if args.run_type == 'test_prompts':
        pass
    elif args.run_type == 'gen_all':
        original_path = args.file_save_path
        #alias = ['_both', '_masked', '_notmasked']
        alias = ['_notmasked']
        alias = ['_both', '_masked',]
        #pts = ["senti_pos_both","senti_pos_masked","senti_pos_notmasked"]
        #pts = ["senti_neg_both","senti_neg_masked","senti_neg_notmasked"]
        #pts = ["nontoxic_both","nontoxic_masked","nontoxic_notmasked"]
        pts = ["nontoxic_notmasked"]
        pts = ["nontoxic_both","nontoxic_masked"]
        for idx, pt in enumerate(pts):
            args.prompt_type = pt
            print("editing for", pt)
            args.file_save_path = original_path.split(".jsonl")[0] + alias[idx] + '.jsonl'
            _gen_hf_or_vllm(args)
    elif args.run_type == 'gen_all_gpt':
        original_path = args.file_save_path
        alias = ['_both', '_masked', '_notmasked']
        pts = ["senti_pos_both","senti_pos_masked","senti_pos_notmasked"]
        for idx, pt in enumerate(pts):
            args.prompt_type = pt
            args.file_save_path = original_path.split(".jsonl")[0] + alias[idx] + '.jsonl'
            generate_and_save_result_gpt(args)
    elif args.hf_model_name.startswith('gpt'):
        print("running generation with gpt")
        generate_and_save_result_gpt(args)
    else:
        _gen_hf_or_vllm(args)
    
    
    