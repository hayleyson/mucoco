###############################################################################
# [파일의 가정]
# SC Energy는 실험을 해볼 부분이 있습니다.
# LLM 한테 넘겨줄 때는 EBM한테 넘겨줄 때와 형식을 다르게 할지 같게 할지!
# 사실 중요한 거 아니에요..
# 제가 생각을 좀 복잡하게 하잖아요? 
# 이렇게 하려고요. 기존의 코드처럼 iteration 별로 locate한 결과 저장할 거고, edit한 결과도 저장할 거에요.
# 저장할 때에는 ebm 형식 text로 저장할 거에요. 
# llm 한테 넘겨주는 부분에서만 형식을 바꿔서 넘겨줄 거에요.
###############################################################################

import os, sys, argparse, time, json, math, re, random, yaml, transformers, torch, huggingface_hub
huggingface_token = os.getenv("HF_TOKEN")
huggingface_hub.login(token=huggingface_token)

from typing import List
from copy import deepcopy
from argparse import Namespace
import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from torch.utils.data import DataLoader, Dataset

from new_module.dev_utils.utils import load_sc_energy_model, parse_set_text, load_eval2_dataset
from new_module.locate.new_locate_utils import LocateMachine4SCE, LocateMachine
from new_module.new_mlm_reranking_all import call_locate, union_masks
from new_module.loc_edit_llm_iter import evaluate_toxicity_losses
from new_module.set_consistency_energy.energynets.decomposition.no_decomposition import no_decomposition_loader
from new_module.set_consistency_energy.energynets.energynet import energynet
import new_module.losses as lossbuilder
from new_module.llm_experiments.edit_with_llm.edit.llm_generate_jsonl import (
    generate_and_save_result_sc_energy,
    generate_and_save_result_vllm_sc_energy,
)

###############################################################################


def locate_modes_for_energy_models(task: str, n_energy: int) -> List[str]:
    """Align with ``new_mlm_reranking_all``: multi-attribute ``task`` uses ``_`` splits; else repeat."""
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
    elif locate_task in ["set_lconvqa", "set_snli"]:
        model_config = yaml.load(open(path, 'r'), 
                                Loader=yaml.FullLoader)
        model_config['device'] = device
        model_e = load_sc_energy_model(path, device)
        return 
    else:       
        clf = AutoModelForSequenceClassification.from_pretrained(path)
        tok = AutoTokenizer.from_pretrained(path)
        clf = clf.to(device)
        clf.eval()
        return LocateMachine(clf, tok, locate_task)



def locate_texts_multi(
    locators: List[LocateMachine],
    locate_modes_list: List[str],
    label_ids: List[int],
    union_tokenizer: AutoTokenizer,
    data_loader: DataLoader,
    output_file: str,
    locate_edit_idx,
    locate_config: dict
) -> List[List[str]]:
    """Locate with every energy model; union [MASK] positions (when len > 1)."""
    print("locate output file path:", output_file)
    print("Locating Start... (%d energy model(s))" % len(locators))

    masked_set_texts = []
    with open(output_file, "w", encoding="utf-8") as outfile:
        for sample_idx, batch in enumerate(data_loader):
            prompt = ""
            print(f"locate_edit_idx: {locate_edit_idx[sample_idx]}")
            if locate_edit_idx[sample_idx][0]:
                print(f"batch: {batch}")
                running = [batch[0][0]]
                if len(locators) == 1:
                    masked_out = call_locate(
                        locate_modes_list[0],
                        label_ids[0],
                        locators[0],
                        prompt,
                        running,
                        locate_config,
                    )
                    merged_text = masked_out[0][0]
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
                    merged_text = union_masks(per_model_masked, union_tokenizer)[0][0]
                masked_set_texts.append(parse_set_text(merged_text, source_mode="ebm"))
                print(f"merged_text: {merged_text}")
                
                outfile.write(json.dumps({"prompt": {"text": ""},
                                    "generations": [{"text": merged_text}]}, ensure_ascii=False) + '\n')
                outfile.flush()
    return masked_set_texts


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def _evaluate_toxicity_losses_single(
    source_text: str,
    hypotheses: list,
    model_e: energynet,
    tokenizer_e: AutoTokenizer,
    target_label_id: int,
    loss_name: str,
    eval_task: str,
) -> tuple:

    class dummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    build_loss_args = dummyArgs(**BUILD_LOSS_DICT)
    build_loss_args.task = eval_task
    threshold = model_e.threshold
    loss_fn = lossbuilder.build_loss(loss_name, model_e, tokenizer_e, build_loss_args)

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
        satisfies_threshold.append(mean_loss <= threshold)

    return losses, satisfies_threshold

def evaluate_toxicity_losses(
    premise: str,
    hypotheses: list,
    model_e_list: list,
    tokenizer_e_list: list,
    label_ids: list,
    loss_names: list,
    eval_tasks: list,
) -> tuple:
    """All energy models must pass; returns (per_model_losses, [all_pass])."""
    losses, sat = _evaluate_toxicity_losses_single(
        premise,
        hypotheses,
        model_e_list[0],
        tokenizer_e_list[0],
        label_ids[0],
        loss_names[0],
        eval_tasks[0],
    )
    return [losses[0]], [sat[0]]



###############################################################################

def contains_chinese(text):
    """
    Check if the given text contains any Chinese characters using regex.
    """
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_char_pattern.search(text))




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


if __name__ == "__main__":
    
    
    parser_main = argparse.ArgumentParser(description="Run experiments with configurable arguments.")

    parser_main.add_argument("job_id", type=str, help="Job ID for the experiment.")
    parser_main.add_argument("--exp_label", type=str,  required=True, help="Experiment label.")
    parser_main.add_argument("--total_iteration", type=int,  required=True, help="Iteration number")

    parser_main.add_argument("--directory", type=str, required=True, help="Base directory for input and output files.")
    parser_main.add_argument(
        "--pretrained_model_path",
        type=str,
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more energy model paths (space-separated). All are used for locate (masks unioned); order aligns with --label_id / --threshold / --losses.",
    )
    parser_main.add_argument("--hf_model_name", type=str, required=True, help="Name of the Hugging Face model.")
    parser_main.add_argument(
        "--use_incon_samples",
        action="store_true",
        help="Subsample locate loaders to inconsistent set instances only.",
    )
    parser_main.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="When --use_incon_samples is set, number of inconsistent samples to draw.",
    )

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
        "--losses",
        type=str,
        nargs="+",
        required=True,
        metavar="LOSS",
        help="Registered loss name per energy model (see new_module.losses; same count as --pretrained_model_path).",
    )
    parser_main.add_argument("--max_num_tokens", type=int, default=7, help="Max number of tokens to locate.")
    parser_main.add_argument(
        "--use_vllm",
        action="store_true",
        help="Run LLM generation with vLLM (faster on GPU; requires vllm package).",
    )

    args_main = parser_main.parse_args()

    # args main parsing done
    ###############################################################################

    use_incon_samples = args_main.use_incon_samples
    n_samples_cfg = args_main.n_samples

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

    energy_loss_names = args_main.losses
    if len(energy_loss_names) != n_energy:
        parser_main.error(
            f"Expected {n_energy} --losses value(s) for {n_energy} --pretrained_model_path(s); "
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ####################################################################################
    
    # Load model config, model, tokenizer
    model_config = yaml.load(open(pretrained_model_paths[0]), Loader=yaml.FullLoader)
    model_config['device'] = device
    model_e = load_sc_energy_model(pretrained_model_paths[0], device)
    mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model_e.representation_model.tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})
    tokenizer_e = model_e.representation_model.tokenizer
    
    # Define LocateMachine
    energy_locators = [LocateMachine4SCE(model_config, model_e, task)]
    locate_modes = [task]
    
    locate_run_config = {
        "device": device,
        "locate_method": locate_option,
        "num_edit_token_per_step": max_num_tokens,
        "locate_unit": "word",
    }

    # Load dataset
    test_datasets, test_steps_names = load_eval2_dataset(task, split="test", use_only_incon=True, n_samples=n_samples_cfg, random_seed=42)
    # set list of list to keep track of which idx to locate & edit
    locate_edit_idx = [[True]] * len(test_datasets[0].dataset)

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
        data_loader = no_decomposition_loader(test_dataset, tokenizer=tokenizer_e, params=model_config).get_loader()
        masked_set_texts = locate_texts_multi(
            energy_locators,
            locate_modes,
            energy_label_ids,
            mlm_tokenizer,
            data_loader,
            locate_output_file_path + f"_filtered_{iter_idx}",
            locate_edit_idx,
            locate_run_config,
        )
        located_dataset = deepcopy(test_dataset)
        located_dataset.dataset = masked_set_texts


        end_time = time.time()
        locate_time = end_time - start_time

        ###############################################################################
        print("start LLM edit")
        start_time = time.time()

        args_dict = {
        'hf_model_name': hf_model_name,
        'file_save_path': edit_output_file_path + f"_filtered_{iter_idx}",
        'input_file_path': locate_output_file_path + f"_filtered_{iter_idx}",
        'locate_edit_idx': locate_edit_idx,  # 리스트 그대로 전달
        'prompt_type': prompt_type,
        'num_return_sequences': 1,
        'max_tokens': 150,
        'enable_thinking': True if 'qwen3' in hf_model_name.lower() else False,
        'dataset': task
    }

        args = Namespace(**args_dict)
        if use_vllm:
            edited_set_texts = generate_and_save_result_vllm_sc_energy(args, test_dataset, located_dataset)
        else:
            edited_set_texts = generate_and_save_result_sc_energy(args, test_dataset, located_dataset)
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
        previous_set_texts = test_dataset.dataset
        modified_set_texts = []

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
            combined_set_texts = []

            for gen_idx, generation in enumerate(prev_generations):
                if locate_edit_idx[line_idx][gen_idx]:  # Use LLM output if True
                    if llm_generations:
                        next_generation = llm_generations.pop(0)
                        next_set_text = edited_set_texts[line_idx]
                        if contains_chinese(next_generation['text']):
                            combined_generations.append(generation)  # Fallback to previous data
                            combined_set_texts.append(previous_set_texts[line_idx])
                        else:
                            combined_generations.append(next_generation)
                            combined_set_texts.append(next_set_text)
                    else:
                        print(f"Warning: No more LLM generations for line {line_idx}, gen {gen_idx}")
                        combined_generations.append(generation)  # Fallback to previous data
                        combined_set_texts.append(previous_set_texts[line_idx])
                else:  # Use previous data if False
                    combined_generations.append(generation)
                    combined_set_texts.append(previous_set_texts[line_idx])

            # Append the updated item to modified_data
            modified_data.append({"prompt": {"text": prompt}, "generations": combined_generations})
            modified_set_texts.extend(combined_set_texts)
        
        # Write combined data to new `total` file
        with open(edit_output_file_path + f"_total_{iter_idx}", 'w', encoding='utf-8') as output_file:
            for item in modified_data:
                json.dump(item, output_file, ensure_ascii=False)
                output_file.write('\n')
        
        # save combined set texts to test_dataset
        test_dataset.dataset = modified_set_texts

        ###############################################################################
        
        print("start eval")
        start_time = time.time()

        
        # L&E text (this iteration)
        l_e_texts = []
        with open(edit_output_file_path + f"_total_{iter_idx}", 'r', encoding='utf-8') as hyps_file:
            for line in hyps_file:
                data = json.loads(line)
                l_e_texts.extend([g['text'] for g in data['generations']])
                
        # Per flat index: premise for NLI / nli_toxicity; else hypothesis text (legacy).
        eval_premises = ["" for _ in range(l_e_texts)]

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
                        [model_e],
                        [tokenizer_e],
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