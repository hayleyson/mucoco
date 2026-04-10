import os
import sys
huggingface_token = os.getenv("HF_TOKEN")

import argparse

###############################################################################
# args

parser_main = argparse.ArgumentParser(description="Run experiments with configurable arguments.")

parser_main.add_argument("job_id", type=str, help="Job ID for the experiment.")
parser_main.add_argument("--exp_label", type=str,  required=True, help="Experiment label.")
parser_main.add_argument("--total_iteration", type=int,  required=True, help="Iteration number")

parser_main.add_argument("--directory", type=str, required=True, help="Base directory for input and output files.")
parser_main.add_argument("--input_file_path", type=str, required=True, help="Path to the input JSONL file.")
parser_main.add_argument("--orig_text_path", type=str, required=True, help="Path to the original text JSONL file.")
parser_main.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained model.")
parser_main.add_argument("--hf_model_name", type=str, required=True, help="Name of the Hugging Face model.")
parser_main.add_argument("--prompt_type", type=str,  required=True, help="Type of the prompt.")
parser_main.add_argument("--task", type=str,  required=True, help="Task type.")
parser_main.add_argument("--label_id", type=int,  required=True, help="Label ID for the task.")
parser_main.add_argument("--locate_option", type=str,  required=True, help="Locate option.")
parser_main.add_argument("--threshold", type=float,  required=True, help="Threshold value.")
parser_main.add_argument("--max_num_tokens", type=int, default=7, help="Max number of tokens to locate.")

args_main = parser_main.parse_args()

# Print received arguments for debugging
print(f"Received arguments: {args_main}")

job_id = args_main.job_id
exp_label = args_main.exp_label
total_iteration = args_main.total_iteration

directory = args_main.directory
input_file_path = args_main.input_file_path
orig_text_path = args_main.orig_text_path
pretrained_model_path = args_main.pretrained_model_path
hf_model_name = args_main.hf_model_name
prompt_type = args_main.prompt_type
task = args_main.task
label_id = args_main.label_id
locate_option = args_main.locate_option
threshold = args_main.threshold
max_num_tokens = args_main.max_num_tokens

for subdir in ['located', 'edited', 'losses', 'final']:
    os.makedirs(directory + '/' + subdir, exist_ok=True)

locate_output_file_path = directory + f'/located/{exp_label}_located_{job_id}.jsonl'
edit_output_file_path = directory + f'/edited/{exp_label}_edited_{job_id}.jsonl'
eval_output_file_path = directory + f'/losses/{exp_label}_losses_{job_id}.txt'
final_output_file_path = directory + f'/final/{exp_label}_loc_edit_{job_id}.jsonl'
energy_model_path = pretrained_model_path + '/'


###############################################################################

# import
import time
import json
import math
import re

import transformers
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader

from new_module.em_training.nli.models import EncoderModel  
from new_module.locate.new_locate_utils import LocateMachine

import new_module.losses as lossbuilder

import huggingface_hub
from argparse import Namespace

###############################################################################
###############################################################################

# locate - llm edit (gpt) - eval (early stopping) iteration


# get ready for locate
    # 환경 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델과 토크나이저 로드
if task == "nli":
    # config
    with open(os.path.join(pretrained_model_path, 'config.json')) as f:
        model_config = json.load(f)
    model_config['device'] = device
    model_config['model_path'] = os.path.join(pretrained_model_path, 'best_model_pearsonr.pth')
    if locate_option == "attention":
        model_config['locate']['type'] = "attention"
    elif locate_option == "grad_norm":
        model_config['locate']['type'] = "gradnorm"
    # load model
    model = EncoderModel(params=model_config)
    model.load_state_dict(torch.load(model_config['model_path'], weights_only=True), strict=False)
    model.eval()
    model.to(device)

    tokenizer = model.tokenizer
else:
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = model.to(device)

# locate에서는 파일, 모델, locate_edit_idx를 받아서
# 이 idx=True 인 경우만 모아 output_file에 저장한다 
def locate_texts(model, tokenizer, input_file, output_file, task, label_id, locate_edit_idx, locate_method, max_num_tokens=7):
    """
    Locates tokens in texts using the provided model and task.
    """

    # LocateMachine 초기화
    locator = LocateMachine(model, tokenizer, task)

    print("locate input file path:", input_file)
    print("locate output file path:", output_file)

    print("Locating Start...")

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line_idx, line in enumerate(infile):
            # JSON 형식으로 변환
            data = json.loads(line)
            prompt = data['prompt']['text']
            generations = data['generations']
            
            masked_generations = []
            # generations 내의 각 text에 대해 LocateMachine 적용
            for gen_idx, generation in enumerate(generations):
                if locate_edit_idx[line_idx][gen_idx]:
                    text = f"<s>{prompt}</s>{generation['text']}</s>" if task == "nli" else generation['text']
                    # locate_main 적용
                    masked_text = locator.locate_main([text], 
                                                        locate_method, 
                                                        max_num_tokens=max_num_tokens, 
                                                        unit='word', 
                                                        label_id=label_id,
                                                        num_layer=10,)
                    # masked 결과를 generation에 추가 (기존 key나 새로운 key 사용 가능)
                    generation['text'] = masked_text[0]  # locate_main은 리스트를 반환하므로 첫 번째 값 선택
                    masked_generations.append(generation)
            if masked_generations:
                # 결과를 다시 JSON 형식으로 변환하고 출력 파일에 쓰기
                data['generations'] = masked_generations
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')



###############################################################################
# get ready for llm edit

huggingface_hub.login(token=huggingface_token)

from new_module.llm_experiments.edit_with_llm.edit.llm_generate_jsonl import generate_and_save_result, generate_and_save_result_gpt

parser = argparse.ArgumentParser()
parser.add_argument('--hf_model_name', type=str)
parser.add_argument('--file_save_path', type=str)
parser.add_argument('--input_file_path', type=str)
parser.add_argument('--orig_text_path', type=str)
parser.add_argument('--locate_edit_idx', type=list, default=[])
parser.add_argument('--prompt_type', type=str)
parser.add_argument('--num_return_sequences', type=int, default=10)
parser.add_argument('--max_tokens', type=int, default=30)


###############################################################################
# get ready for eval

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def evaluate_toxicity_losses(source_text: str, hypotheses: list, config: dict, threshold: float) -> tuple:
    """
    Evaluate toxicity losses for given hypotheses and determine if losses satisfy a given threshold.
    
    Parameters:
        source_text (str): Original source text.
        hypotheses (list): List of hypothesis texts to evaluate.
        config (dict): Configuration dictionary with model, tokenizer, and loss settings.
        threshold (float): Loss threshold to determine satisfaction.

    Returns:
        losses (list): List of loss values for each hypothesis.
        satisfies_threshold (list): List of booleans indicating whether each loss satisfies the threshold.
    """

    # Load tokenizer, model, and define loss
    # use only [1] bc [0] is not used for early stopping
    if task == 'nli':
        with open(os.path.join(energy_model_path, 'config.json')) as f:
            config_m = json.load(f)
        config_m['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_path = os.path.join(energy_model_path, 'best_model_pearsonr.pth')
        config_m['model_path'] = model_path

        ## load model
        model = EncoderModel(config_m)
        model = model.to(config_m['device'])
        model.load_state_dict(torch.load(model_path,weights_only=True),strict=False)
        model.eval()
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            energy_model_path, cache_dir=config["cache_dir"], use_fast=True
        )
        model_config = AutoConfig.from_pretrained(energy_model_path, cache_dir=config["cache_dir"])
        model = lossbuilder.ModelWrapper(
            AutoModelForSequenceClassification.from_pretrained(
                energy_model_path, config=model_config, cache_dir=config["cache_dir"]
            )
        )
        model.eval().to(config["device"])

    # Build losses
    class dummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    build_loss_args = dummyArgs(**config["build_loss_dict"])
    build_loss_args.task = config["task"]

    if task == 'nli':
        loss_fn = lossbuilder.build_loss(
            "classification", model, tokenizer, build_loss_args
        )
    else:
        loss_fn = lossbuilder.build_loss(
            config["losses"][1], model, tokenizer, build_loss_args
        )

    # Compute losses for each hypothesis
    losses = []
    satisfies_threshold = []
    threshold_log = -math.log(threshold)  # Convert threshold to log scale

    for hypothesis in hypotheses:
        curr_loss = []
        data_loader = DataLoader(CustomDataset(hypothesis), batch_size=config.get('batch_size', 64))

        with torch.no_grad():
            for batch in data_loader:
                loss_values = loss_fn.compute_gold_loss(
                    source_text, batch,
                    label_id=config['target_label_ids'][1],  # For toxicity
                )
                curr_loss.extend(loss_values.cpu().tolist())
                torch.cuda.empty_cache()

        # Calculate mean loss and check threshold satisfaction
        mean_loss = torch.tensor(curr_loss).mean().item()
        losses.append(mean_loss)
        satisfies_threshold.append(mean_loss < threshold_log)

    return losses, satisfies_threshold

config = {
    "model_paths": [
        "gpt2-large", 
        energy_model_path
    ],
    "tokenizer_paths": [
        "gpt2-large", 
        energy_model_path
    ],
    "model_types": ["AutoModelForCausalLM", "AutoModelForSequenceClassification"],
    "losses": ["gpt2", "classification_no_prefix_logprobloss"],
    "build_loss_dict": {
        "coeff_steps": 200,
        "coeff_pattern": "constant",
        "loss_type": "xentropy",
        "length_normalize": False,
        "AR_temperature": 1.0,
        "AR_top_k": 0,
        "AR_top_p": 0.96,
        "max_output_length": 20
    },
    "task": task,
    "device": "cuda",
    "cache_dir": "/home/hyeryung/data/.cache", # Change to your huggingface cache directory
    "batch_size": 64,
    "target_label_ids": [None, label_id],  # Example target labels
}



###############################################################################

def contains_chinese(text):
    """
    Check if the given text contains any Chinese characters using regex.
    """
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_char_pattern.search(text))

# set list of list to keep track of which idx to locate & edit
locate_edit_idx = []
#for i in range(250):
    #locate_edit_idx.append([True] * 10)
with open(input_file_path, 'r', encoding='utf-8') as infile:
# 출력 파일 열기
    for line_idx, line in enumerate(infile):
        # JSON 형식으로 변환
        data = json.loads(line)
        generations = data['generations']
        locate_edit_idx.append([True] * len(generations))

modified_data = []
# start iteration
for iter_idx in range(total_iteration):

    need_locate_count = sum(sum(row) for row in locate_edit_idx)
    if need_locate_count == 0:
        print("All data satisfied!")
        break
    print(f"ITERATION #{iter_idx}: {need_locate_count} items need processing.")

    print("###############################################################################")
    print(f"ITERATION #{iter_idx}")
    iter_start_time = time.time()
    filtered_data = []  # LLM generation input


    print("start locate")
    start_time = time.time()
    locate_texts(model, 
                 tokenizer, 
                 input_file_path, 
                 locate_output_file_path + f"_filtered_{iter_idx}", 
                 task,
                 label_id,
                 locate_edit_idx,
                 locate_option,
                 max_num_tokens=max_num_tokens
                 )


    end_time = time.time()
    locate_time = end_time - start_time
    print(f"LOCATE TIME #{iter_idx}: {locate_time}")

    ###############################################################################
    print("start LLM edit")
    start_time = time.time()

    args_dict = {
    'hf_model_name': hf_model_name,
    'file_save_path': edit_output_file_path + f"_filtered_{iter_idx}",
    'input_file_path': locate_output_file_path + f"_filtered_{iter_idx}",
    'orig_text_path': orig_text_path,
    'locate_edit_idx': locate_edit_idx,  # 리스트 그대로 전달
    'prompt_type': prompt_type,
    'num_return_sequences': 1,
    'max_tokens': 150,
}

    args = Namespace(**args_dict)
    generate_and_save_result(args)
    end_time = time.time()
    edit_time = end_time - start_time
    print(f"LLM EDIT TIME #{iter_idx}: {edit_time}")

    
    ###############################################################################
    print("start combining previous and new data")

    # Load previous `_total` data if not first iter_idxation
    previous_data = []
    if iter_idx > 0:
        with open(edit_output_file_path + f"_total_{iter_idx - 1}", 'r', encoding='utf-8') as prev_file:
            for line in prev_file:
                previous_data.append(json.loads(line))
    else:
        with open(edit_output_file_path + f"_filtered_{iter_idx}", 'r', encoding='utf-8') as prev_file:
            for line in prev_file:
                previous_data.append(json.loads(line))      

    modified_data = []

    with open(edit_output_file_path + f"_filtered_{iter_idx}", 'r', encoding='utf-8') as llm_file:
        llm_generations = []
        for llm_line in llm_file:
            llm_data = json.loads(llm_line)
            llm_generations.extend(llm_data["generations"])  # Flatten into 1D list
    # Combine previous data with flattened LLM generations
    for line_idx, prev_item in enumerate(previous_data):
        prompt = prev_item["prompt"]["text"]
        prev_generations = prev_item["generations"]
        combined_generations = []

        for gen_idx, generation in enumerate(prev_generations):
            if locate_edit_idx[line_idx][gen_idx]:  # Use LLM output if True
                if llm_generations:
                    next_generation = llm_generations.pop(0)
                    if contains_chinese(next_generation['text']):
                        combined_generations.append(generation)  # Fallback to previous data
                    else:
                        combined_generations.append(next_generation)
                else:
                    print(f"Warning: No more LLM generations for line {line_idx}, gen {gen_idx}")
                    combined_generations.append(generation)  # Fallback to previous data
            else:  # Use previous data if False
                combined_generations.append(generation)

        # Append the updated item to modified_data
        modified_data.append({"prompt": {"text": prompt}, "generations": combined_generations})
    
    # Write combined data to new `total` file
    with open(edit_output_file_path + f"_total_{iter_idx}", 'w', encoding='utf-8') as output_file:
        for item in modified_data:
            json.dump(item, output_file, ensure_ascii=False)
            output_file.write('\n')

    ###############################################################################
    
    print("start eval")
    start_time = time.time()

    # data loading
    # source text (before locate & edit)
    source_texts = []

    start_time = time.time()
    with open(input_file_path, 'r', encoding='utf-8') as infile:
    # 출력 파일 열기
        if task == 'nli':
            for line_idx, line in enumerate(infile):
                # JSON 형식으로 변환
                data = json.loads(line)
                premise = data['prompt']['text']
                generations = data['generations']
                for gen_idx, generation in enumerate(generations):
                    source_texts.append(premise)
        else: 
            for line_idx, line in enumerate(infile):
                # JSON 형식으로 변환
                data = json.loads(line)
                generations = data['generations']
                for gen_idx, generation in enumerate(generations):
                    text = generation['text']
                    source_texts.append(text)

    # L&E text (this iteration)
    l_e_texts = []
    with open(edit_output_file_path + f"_total_{iter_idx}", 'r', encoding='utf-8') as hyps_file:
        for line in hyps_file:
            data = json.loads(line)
            l_e_texts.extend([g['text'] for g in data['generations']])

    row_idx = 0
    col_idx = 0
    with open(eval_output_file_path + f"_{iter_idx}", 'w', encoding='utf-8') as f:
        for src_idx, source_text in enumerate(source_texts):
            if locate_edit_idx[row_idx][col_idx]:
                l_e_text = l_e_texts[src_idx]
                losses, satisfies = evaluate_toxicity_losses(source_text, [[l_e_text]], config, threshold)
                f.write(f"{row_idx},{col_idx},{losses[0]},{satisfies[0]}\n")
                if satisfies[0]:
                    locate_edit_idx[row_idx][col_idx] = False
            if len(locate_edit_idx[row_idx]) == col_idx + 1:
                row_idx += 1
                col_idx = 0
            else:
                col_idx += 1
    end_time = time.time()
    eval_time = end_time - start_time
    print(f"EVAL TIME #{iter_idx}: {eval_time}")

    input_file_path = edit_output_file_path + f"_total_{iter_idx}"
    print("END OF ITERATION, file directory updated.")
    iter_end_time = time.time()
    print(f"TIME TAKEN: {(iter_end_time- iter_start_time)/60} mins")
    print("input_file_path", input_file_path)




###############################################################################
###############################################################################


# print iteration
print("###############################################################################")
print("total iteration:", iter_idx+1)
need_locate_count = sum(sum(row) for row in locate_edit_idx)
if need_locate_count == 0:
    print("All data satisfied!")
else:
    print(f"ITERATION #{iter_idx}: {need_locate_count} items need processing.")
# save to final
with open(final_output_file_path, 'w', encoding='utf-8') as output_file:
    for item in modified_data:
        json.dump(item, output_file, ensure_ascii=False)
        output_file.write('\n')
print("saved final result to:", final_output_file_path)