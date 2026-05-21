import os
import sys
huggingface_token = os.getenv("HF_TOKEN")

import argparse
from typing import List

###############################################################################
# args

parser_main = argparse.ArgumentParser(description="Run experiments with configurable arguments.")

parser_main.add_argument("job_id", type=str, help="Job ID for the experiment.")
parser_main.add_argument("--exp_label", type=str,  required=True, help="Experiment label.")
parser_main.add_argument("--total_iteration", type=int,  required=True, help="Iteration number")

parser_main.add_argument("--directory", type=str, required=True, help="Base directory for input and output files.")
parser_main.add_argument("--input_file_path", type=str, required=True, help="Path to the input JSONL file.")
parser_main.add_argument("--orig_text_path", type=str, required=True, help="Path to the original text JSONL file.")
parser_main.add_argument(
    "--pretrained_model_path",
    type=str,
    nargs="+",
    required=True,
    metavar="PATH",
    help="One or more energy model paths (space-separated). All are used for locate (masks unioned); order aligns with --label_id / --threshold / --loss_name.",
)
parser_main.add_argument("--hf_model_name", type=str, required=True, help="Name of the Hugging Face model.")
parser_main.add_argument("--prompt_type", type=str,  required=True, help="Type of the prompt.")
parser_main.add_argument("--task", type=str,  required=True, help="Task type.")
parser_main.add_argument(
    "--label_id",
    type=int,
    nargs="+",
    required=True,
    metavar="ID",
    help="Target label ids (space-separated), one per energy model (same count as --pretrained_model_path).",
)
parser_main.add_argument("--locate_option", type=str,  required=True, help="Locate option.")
parser_main.add_argument(
    "--threshold",
    type=float,
    nargs="+",
    required=True,
    metavar="THR",
    help="Thresholds (space-separated), one per energy model (same count as --pretrained_model_path).",
)
parser_main.add_argument(
    "--loss_name",
    type=str,
    nargs="+",
    required=True,
    metavar="LOSS",
    help="Registered loss name per energy model (see new_module.edit.ebm.losses; same count as --pretrained_model_path).",
)
parser_main.add_argument("--max_num_tokens", type=int, default=7, help="Max number of tokens to locate.")
parser_main.add_argument(
    "--use_vllm",
    action="store_true",
    help="Run LLM generation with vLLM (faster on GPU; requires vllm package).",
)

args_main = parser_main.parse_args()

pretrained_model_paths = args_main.pretrained_model_path
energy_thresholds = args_main.threshold
n_energy = len(pretrained_model_paths)
if len(energy_thresholds) != n_energy:
    parser_main.error(
        f"Expected {n_energy} --threshold value(s) for {n_energy} --pretrained_model_path(s); "
        f"got {len(energy_thresholds)}."
    )

energy_label_ids = args_main.label_id
if len(energy_label_ids) != n_energy:
    parser_main.error(
        f"Expected {n_energy} --label_id value(s) for {n_energy} --pretrained_model_path(s); "
        f"got {len(energy_label_ids)}."
    )

energy_loss_names = args_main.loss_name
if len(energy_loss_names) != n_energy:
    parser_main.error(
        f"Expected {n_energy} --loss_name value(s) for {n_energy} --pretrained_model_path(s); "
        f"got {len(energy_loss_names)}."
    )

# Print received arguments for debugging
print(f"Received arguments: {args_main}")
print(f"energy_models ({n_energy}): {pretrained_model_paths}")
print(f"energy_thresholds: {energy_thresholds}")
print(f"energy_label_ids: {energy_label_ids}")
print(f"energy_loss_names: {energy_loss_names}")

job_id = args_main.job_id
exp_label = args_main.exp_label
total_iteration = args_main.total_iteration

directory = args_main.directory
input_file_path = args_main.input_file_path
orig_text_path = args_main.orig_text_path
hf_model_name = args_main.hf_model_name
prompt_type = args_main.prompt_type
task = args_main.task
locate_option = args_main.locate_option
max_num_tokens = args_main.max_num_tokens
use_vllm = args_main.use_vllm

for subdir in ['located', 'edited', 'losses', 'final']:
    os.makedirs(directory + '/' + subdir, exist_ok=True)

locate_output_file_path = directory + f'/located/{exp_label}_located_{job_id}.jsonl'
edit_output_file_path = directory + f'/edited/{exp_label}_edited_{job_id}.jsonl'
eval_output_file_path = directory + f'/losses/{exp_label}_losses_{job_id}.txt'
final_output_file_path = directory + f'/final/{exp_label}_loc_edit_{job_id}.jsonl'
time_log_path = final_output_file_path + ".time"


###############################################################################

# import
import multiprocessing as mp

# vLLM V1 starts engine workers via multiprocessing; default "fork" breaks if the parent
# already touched CUDA (e.g. torch.cuda.is_available()). "spawn" gives workers a clean runtime.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import time
import json
import math
import re

import transformers
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader

from new_module.utils.utils import load_sc_energy_model
from new_module.ebm_training.nli.models.encoder import EncoderModel  
from new_module.locate.ebm.locate_utils import LocateMachine
from new_module.locate.ebm.locate_utils import call_locate, union_masks

import new_module.edit.ebm.losses as lossbuilder

import huggingface_hub
from argparse import Namespace

###############################################################################
###############################################################################

# locate - llm edit (gpt) - eval (early stopping) iteration


def locate_modes_for_energy_models(task: str, n_energy: int) -> List[str]:
    """Align with ``ebm_edit_main``: multi-attribute ``task`` uses ``_`` splits; else repeat."""
    if n_energy == 1:
        return [task]
    parts = task.split("_")
    if len(parts) == n_energy:
        return parts
    return [task] * n_energy

def load_locate_machine_for_path(
    path: str,
    locate_task: str,
    locate_option: str,
    device: str,
) -> LocateMachine:
    """One energy checkpoint + LocateMachine for that path's locate mode."""
    if locate_task == "nli":
        with open(os.path.join(path, "config.json")) as f:
            model_config = json.load(f)
        model_config["device"] = device
        model_config["model_path"] = os.path.join(path, "best_model_pearsonr.pth")
        if locate_option == "attention":
            model_config["locate"]["type"] = "attention"
        elif locate_option == "grad_norm":
            model_config["locate"]["type"] = "gradnorm"
        enc_model = EncoderModel(params=model_config)
        enc_model.load_state_dict(
            torch.load(model_config["model_path"], weights_only=True), strict=False
        )
        enc_model.eval()
        enc_model.to(device)
        return LocateMachine(enc_model, enc_model.tokenizer, locate_task)

    clf = AutoModelForSequenceClassification.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path)
    clf = clf.to(device)
    clf.eval()
    return LocateMachine(clf, tok, locate_task)


device = "cuda" if torch.cuda.is_available() else "cpu"
locate_modes = locate_modes_for_energy_models(task, n_energy)
print(f"locate_modes ({len(locate_modes)}): {locate_modes}")

mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
energy_locators: List[LocateMachine] = []
for i in range(n_energy):
    loc = load_locate_machine_for_path(
        pretrained_model_paths[i],
        locate_modes[i],
        locate_option,
        device,
    )
    loc.tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})
    energy_locators.append(loc)

locate_run_config = {
    "device": device,
    "locate_method": locate_option,
    "num_edit_token_per_step": max_num_tokens,
    "locate_unit": "word",
}


def locate_texts_multi(
    locators: List[LocateMachine],
    locate_modes_list: List[str],
    label_ids: List[int],
    union_tokenizer: AutoTokenizer,
    input_file: str,
    output_file: str,
    locate_edit_idx,
    locate_config: dict,
):
    """Locate with every energy model; union [MASK] positions (when len > 1)."""
    print("locate input file path:", input_file)
    print("locate output file path:", output_file)
    print("Locating Start... (%d energy model(s))" % len(locators))

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line_idx, line in enumerate(infile):
            data = json.loads(line)
            prompt = data["prompt"]["text"]
            generations = data["generations"]

            masked_generations = []
            for gen_idx, generation in enumerate(generations):
                if locate_edit_idx[line_idx][gen_idx]:
                    running = [generation["text"]]
                    if len(locators) == 1:
                        masked_out = call_locate(
                            locate_modes_list[0],
                            label_ids[0],
                            locators[0],
                            prompt,
                            running,
                            locate_config,
                        )
                        merged_text = masked_out[0]
                    else:
                        per_model_masked = []
                        for li, loc in enumerate(locators):
                            per_model_masked.append(
                                call_locate(
                                    locate_modes_list[li],
                                    label_ids[li],
                                    loc,
                                    prompt,
                                    running,
                                    locate_config,
                                )
                            )
                        merged_text = union_masks(per_model_masked, union_tokenizer)[0]
                    generation["text"] = merged_text
                    masked_generations.append(generation)
            if masked_generations:
                # 결과를 다시 JSON 형식으로 변환하고 출력 파일에 쓰기
                data['generations'] = masked_generations
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')



###############################################################################
# get ready for llm edit

huggingface_hub.login(token=huggingface_token)

from new_module.edit.llm.nli_toxicity.llm_generate_jsonl import (
    generate_and_save_result,
    generate_and_save_result_gpt,
    generate_and_save_result_vllm,
)

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


BUILD_LOSS_DICT = {
    "coeff_steps": 200,
    "coeff_pattern": "constant",
    "loss_type": "xentropy",
    "length_normalize": False,
    "AR_temperature": 1.0,
    "AR_top_k": 0,
    "AR_top_p": 0.96,
    "max_output_length": 20,
}
EVAL_DEVICE = "cuda"
EVAL_CACHE_DIR = "/home/hyeryung/data/.cache"
EVAL_BATCH_SIZE = 64


def _evaluate_toxicity_losses_single(
    source_text: str,
    hypotheses: list,
    energy_root: str,
    threshold: float,
    target_label_id: int,
    loss_name: str,
    eval_task: str,
) -> tuple:
    energy_model_path = energy_root.rstrip("/") + "/"

    class dummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    build_loss_args = dummyArgs(**BUILD_LOSS_DICT)
    build_loss_args.task = eval_task

    if eval_task == "nli":
        with open(os.path.join(energy_root, "config.json")) as f:
            config_m = json.load(f)
        config_m["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(energy_root, "best_model_pearsonr.pth")
        config_m["model_path"] = model_path
        model_e = EncoderModel(params=config_m)
        model_e = model_e.to(config_m["device"])
        model_e.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        model_e.eval()
        tokenizer_e = model_e.tokenizer
    elif eval_task in ["set_lconvqa", "set_snli"]:
        model_e = load_sc_energy_model(energy_root, EVAL_DEVICE)
        tokenizer_e = model_e.representation_model.tokenizer
    else:
        tokenizer_e = AutoTokenizer.from_pretrained(
            energy_model_path, cache_dir=EVAL_CACHE_DIR, use_fast=True
        )
        model_config = AutoConfig.from_pretrained(
            energy_model_path, cache_dir=EVAL_CACHE_DIR
        )
        model_e = lossbuilder.ModelWrapper(
            AutoModelForSequenceClassification.from_pretrained(
                energy_model_path, config=model_config, cache_dir=EVAL_CACHE_DIR
            )
        )
        model_e.eval().to(EVAL_DEVICE)

    loss_fn = lossbuilder.build_loss(loss_name, model_e, tokenizer_e, build_loss_args)

    if eval_task in ["set_lconvqa", "set_snli"]:
        threshold_log = threshold
    else:
        threshold_log = -math.log(threshold)

    losses = []
    satisfies_threshold = []
    for hypothesis in hypotheses:
        curr_loss = []
        data_loader = DataLoader(
            CustomDataset(hypothesis), batch_size=EVAL_BATCH_SIZE
        )
        with torch.no_grad():
            for batch in data_loader:
                loss_values = loss_fn.compute_gold_loss(
                    source_text,
                    batch,
                    label_id=target_label_id,
                )
                curr_loss.extend(loss_values.cpu().tolist())
                torch.cuda.empty_cache()
        mean_loss = torch.tensor(curr_loss).mean().item()
        losses.append(mean_loss)
        if eval_task in ["set_lconvqa", "set_snli"]:
            satisfies_threshold.append(mean_loss <= threshold_log)
        else:
            satisfies_threshold.append(mean_loss < threshold_log)

    return losses, satisfies_threshold


def evaluate_toxicity_losses(
    premise: str,
    hypotheses: list,
    energy_roots: list,
    thresholds: list,
    label_ids: list,
    loss_names: list,
    eval_tasks: list,
) -> tuple:
    """All energy models must pass; returns (per_model_losses, [all_pass])."""
    if len(eval_tasks) != len(energy_roots):
        raise ValueError(
            f"eval_tasks length {len(eval_tasks)} != energy_roots {len(energy_roots)}"
        )
    per_model_losses = []
    per_model_sat = []
    for root, thresh, lid, lname, etask in zip(
        energy_roots, thresholds, label_ids, loss_names, eval_tasks
    ):
        losses, sat = _evaluate_toxicity_losses_single(
            premise,
            hypotheses,
            root,
            thresh,
            lid,
            lname,
            etask,
        )
        per_model_losses.append(losses[0])
        per_model_sat.append(sat[0])
    return per_model_losses, [all(per_model_sat)]



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
run_start_time = time.time()
with open(time_log_path, "w", encoding="utf-8") as tf:
    tf.write(f"output_jsonl={final_output_file_path}\n")
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
    locate_texts_multi(
        energy_locators,
        locate_modes,
        energy_label_ids,
        mlm_tokenizer,
        input_file_path,
        locate_output_file_path + f"_filtered_{iter_idx}",
        locate_edit_idx,
        locate_run_config,
    )


    end_time = time.time()
    locate_time = end_time - start_time

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
    'enable_thinking': True if 'qwen3' in hf_model_name.lower() else False,
}

    args = Namespace(**args_dict)
    if use_vllm:
        generate_and_save_result_vllm(args)
    else:
        generate_and_save_result(args)
    end_time = time.time()
    edit_time = end_time - start_time

    
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

    # Per flat index: premise for NLI / nli_toxicity; else hypothesis text (legacy).
    eval_premises = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        if task in ("nli", "nli_toxicity"):
            for line in infile:
                data = json.loads(line)
                premise = data['prompt']['text']
                for _generation in data['generations']:
                    eval_premises.append(premise)
        else:
            for line in infile:
                data = json.loads(line)
                generations = data['generations']
                for generation in generations:
                    eval_premises.append(generation['text'])

    # L&E text (this iteration)
    l_e_texts = []
    with open(edit_output_file_path + f"_total_{iter_idx}", 'r', encoding='utf-8') as hyps_file:
        for line in hyps_file:
            data = json.loads(line)
            l_e_texts.extend([g['text'] for g in data['generations']])

    row_idx = 0
    col_idx = 0
    with open(eval_output_file_path + f"_{iter_idx}", 'w', encoding='utf-8') as f:
        loss_header = ",".join(f"loss_m{i}" for i in range(n_energy))
        f.write(f"row,col,{loss_header},satisfied_all\n")
        for src_idx, premise in enumerate(eval_premises):
            if locate_edit_idx[row_idx][col_idx]:
                l_e_text = l_e_texts[src_idx]
                losses, satisfies = evaluate_toxicity_losses(
                    premise,
                    [[l_e_text]],
                    pretrained_model_paths,
                    energy_thresholds,
                    energy_label_ids,
                    energy_loss_names,
                    locate_modes
                )
                losses_csv = ",".join(str(x) for x in losses)
                f.write(f"{row_idx},{col_idx},{losses_csv},{satisfies[0]}\n")
                if satisfies[0]:
                    locate_edit_idx[row_idx][col_idx] = False
            if len(locate_edit_idx[row_idx]) == col_idx + 1:
                row_idx += 1
                col_idx = 0
            else:
                col_idx += 1
    end_time = time.time()
    eval_time = end_time - start_time

    input_file_path = edit_output_file_path + f"_total_{iter_idx}"
    print("END OF ITERATION, file directory updated.")
    iter_end_time = time.time()
    iteration_elapsed = iter_end_time - iter_start_time
    with open(time_log_path, "a", encoding="utf-8") as tf:
        tf.write(f"\n[iteration {iter_idx}]\n")
        tf.write(f"locate_seconds={locate_time}\n")
        tf.write(f"llm_edit_seconds={edit_time}\n")
        tf.write(f"eval_seconds={eval_time}\n")
        tf.write(f"iteration_elapsed_seconds={iteration_elapsed}\n")
        tf.write(f"iteration_elapsed_minutes={iteration_elapsed / 60.0}\n")
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
with open(time_log_path, "a", encoding="utf-8") as tf:
    tf.write("\n[summary]\n")
    run_elapsed = time.time() - run_start_time
    tf.write(f"run_wall_seconds={run_elapsed}\n")
    tf.write(f"run_wall_minutes={run_elapsed / 60.0}\n")
print("saved final result to:", final_output_file_path)
print("saved timing log to:", time_log_path)