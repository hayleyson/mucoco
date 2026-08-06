import os, json, time, re, random, sys, pickle, argparse

# vLLM starts EngineCore in a child process. Default worker method is "fork", which
# breaks if PyTorch has already initialized CUDA in this process
# ("Cannot re-initialize CUDA in forked subprocess"). Set before importing torch/vLLM.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import torch
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer
import pandas as pd

from laser_edit.edit.llm.set_consistency.llms import GPT, HFModel, VllmModel
from laser_edit.utils.sc_energy_utils import load_sc_energy_dataset
sys.path.append("laser_edit/set_consistency_energy")
from baselines.LLM.lm_loader import lm_loader

# =========================
# set_consistency_dataset 로더 유틸
# =========================


def main():

    parser = argparse.ArgumentParser('')
    parser.add_argument('model_id', type=str)
    parser.add_argument('--dataset_name', type=str, default='lconvqa', choices=['lconvqa', 'set_nli'])
    parser.add_argument('--output_dir', type=str, default=None, required=True)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--reasoning_effort', type=str, default=None, choices=['none', 'low', 'medium', 'high'])
    parser.add_argument('--use_incon_samples', action='store_true')
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--mode', type=str, choices=['wo_locate', 'w_gt_locate', 'w_self_locate', 'w_ebm_locate', 'w_random_locate'], default='wo_locate')
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument(
        '--decoding',
        type=str,
        choices=['auto', 'nucleus', 'greedy'],
        default='auto',
        help="Decoding policy for local HF/vLLM editors. "
             "'auto' uses Qwen3 thinking-mode sampling for Qwen3, else nucleus "
             "(temp=1.0, top_p=0.96, matching toxicity/NLI). "
             "'nucleus' forces temp=1.0 / top_p=0.96. "
             "'greedy' forces do_sample=False / temperature=0.",
    )
    args = parser.parse_args()
    if args.reasoning_effort == 'none':
        args.reasoning_effort = None
    params = dict(
        dataset_path=args.dataset_path,
        dataset=args.dataset_name, # one of ['lconvqa', 'set_nli']
        task='locate',
        baseline=dict(
            type='llm', 
            model=args.model_id,
            shot_num=5,
            locate_type='all_in_one',
            prediction_type='all_in_one',
            reasoning_effort=args.reasoning_effort,
        ),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=1,
    )

    random.seed(args.random_seed)

    # Create output directory if not existent.
    os.makedirs(args.output_dir, exist_ok=True)
    

    if params['dataset_path'] is None:

        dataset_name = params['dataset']
        # load_sc_energy_dataset uses random.sample when n_samples is not None; default -1 means "all".
        n_samples_kw = None if args.n_samples < 0 else args.n_samples
        test_datasets, test_steps_names = load_sc_energy_dataset(
            dataset_name, "test", args.use_incon_samples, n_samples_kw, args.random_seed
        )

    else:
        with open(params['dataset_path'], "rb") as f:
            test_dataset = pickle.load(f)
        test_datasets = [test_dataset]
        test_steps_names = ['incon']

    print("[Dataset count by dataset name]")
    for dataset_name, dataset in zip(test_steps_names, test_datasets):
        print(f"{dataset_name}: {len(dataset)}")

    test_steps = [lm_loader(e, params=params).get_loader() for e in test_datasets]


    # Set model_name for output file name
    model_name = args.model_id.split('/')[-1]
    if args.reasoning_effort is not None:
        model_name += f"_{args.reasoning_effort}"

    # Initialize model
    if 'gpt' in args.model_id.lower():
        model = GPT(args.model_id, 
                    args.reasoning_effort,
                    args.dataset_name)
    else:
        if args.use_vllm:
            model = VllmModel(args.model_id,
                              args.dataset_name,
                              decoding=args.decoding,
                              seed=args.random_seed)
        else:
            model = HFModel(args.model_id,
                            args.dataset_name,
                            decoding=args.decoding,
                            seed=args.random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    located_indexes_dict = {}
    _ds_stem = args.dataset_name.replace("set_", "")
    if args.mode == 'w_self_locate':
        with open(os.path.join(args.output_dir.replace('llm', 'locate'), f"set_{_ds_stem}_{model_name.lower()}_locate_result.jsonl"), 'r') as f:
            for line in f:
                row = json.loads(line.strip())
                located_indexes_dict[row['data_name']] = row['pred']
    elif args.mode == 'w_ebm_locate':
        with open(os.path.join(args.output_dir.replace('llm', 'locate'), f"set_{_ds_stem}_ebm_locate_result.jsonl"), 'r') as f:
            for line in f:
                row = json.loads(line.strip())
                located_indexes_dict[row['data_name']] = row['pred']
    elif args.mode == 'w_random_locate':
        with open(os.path.join(args.output_dir.replace('llm', 'locate'), f"set_{_ds_stem}_random_locate_result.jsonl"), 'r') as f:
            for line in f:
                row = json.loads(line.strip())
                located_indexes_dict[row['data_name']] = row['pred']

    results = {}
    total_time_seconds = 0
    total_reasoning_tokens = 0
    total_completion_tokens = 0
    avg_sc_energies = []
    contradiction_probas = []
    contradiction_probas_gpt = []

    for es_idx, data_loader in enumerate(test_steps):
        data_name = test_steps_names[es_idx]
        
        if (args.mode == 'w_self_locate') or (args.mode == 'w_ebm_locate') or (args.mode == 'w_random_locate'):
            preds_for_data = located_indexes_dict.get(data_name) if located_indexes_dict else None
            if preds_for_data is None:
                raise ValueError(
                    f"Missing locate predictions for data_name={data_name!r} (mode={args.mode}). "
                    "Ensure the locate result JSONL exists under --output_dir and keys match test step names."
                )
            preds_for_data = [sorted(preds) for preds in preds_for_data]
        else:
            preds_for_data = None
        
        outputs = []
        raw_responses = []
        reasoning_tokens = 0
        completion_tokens = 0
        start = time.time()

        if 'gpt' in args.model_id.lower():
            for i, data in enumerate(data_loader):
                if args.mode == 'w_gt_locate':
                    loc_idx = [idx + 1 for idx in data[0][1]]
                elif (args.mode == 'w_self_locate') or (args.mode == 'w_ebm_locate') or (args.mode == 'w_random_locate'):
                    loc_idx = preds_for_data[i]
                elif (args.mode == 'wo_locate'):
                    loc_idx = None

                result = model.edit(data[0][0], located_indexes=loc_idx)

                outputs.append(result['edited_pairs'])
                raw_responses.append(result['raw_response'])
                reasoning_tokens += result['reasoning_tokens']
                completion_tokens += result['total_generated_tokens']

        else:
            input_data_list = []
            loc_idx_list = []
            for i, data in enumerate(data_loader):
                input_data_list.append(data[0][0])
                if args.mode == 'w_gt_locate':
                    loc_idx_list.append([idx + 1 for idx in data[0][1]])
                elif (args.mode == 'w_self_locate') or (args.mode == 'w_ebm_locate') or (args.mode == 'w_random_locate'):
                    loc_idx_list.append(preds_for_data[i])

            if args.mode == 'wo_locate':
                loc_idx_list = None
            
            result = model.edit(input_data_list, located_indexes_list=loc_idx_list)

            outputs = result['edited_pairs_list']
            raw_responses = result['raw_response_list']
            reasoning_tokens = result['total_reasoning_tokens']
            completion_tokens = result['total_completion_tokens']


        time_seconds = time.time() - start
        
        results[f"edited_pairs-{es_idx+1}-{data_name}"]      = outputs
        results[f"raw_response-{es_idx+1}-{data_name}"]      = raw_responses
        results[f"total_reasoning_tokens-{es_idx+1}-{data_name}"]      = reasoning_tokens
        results[f"total_completion_tokens-{es_idx+1}-{data_name}"]      = completion_tokens
        results[f"time_seconds-{es_idx+1}-{data_name}"]      = time_seconds

        total_time_seconds += time_seconds
        total_reasoning_tokens += reasoning_tokens
        total_completion_tokens += completion_tokens

    results["total_time_seconds"] = total_time_seconds
    results["total_reasoning_tokens"] = total_reasoning_tokens
    results["total_completion_tokens"] = total_completion_tokens

    print(f"total_reasoning_tokens:{results['total_reasoning_tokens']}, total_completion_tokens:{results['total_completion_tokens']}")
    print(f"total_time_seconds:{results['total_time_seconds']}")

    # Format predicted and gold inconsistent pair indices to save.
    response_list = []
    raw_response_list = []
    for es_idx, data_loader in enumerate(test_steps):
        data_name = test_steps_names[es_idx]
        response_list.append({
            'data_name': data_name,
            'pred': results[f"edited_pairs-{es_idx+1}-{data_name}"],
        })
        raw_response_list.append({
            'data_name': data_name,
            'raw_response': results[f"raw_response-{es_idx+1}-{data_name}"],
        })

    parsed_results = pd.DataFrame.from_dict(response_list)
    parsed_results.to_json(os.path.join(args.output_dir, f'set_{args.dataset_name.replace("set_", "")}_{model_name.lower()}_{args.mode.lower()}_edit_result.jsonl'), lines=True, orient='records')
    raw_responses = pd.DataFrame.from_dict(raw_response_list)
    raw_responses.to_json(os.path.join(args.output_dir, f'set_{args.dataset_name.replace("set_", "")}_{model_name.lower()}_{args.mode.lower()}_edit_raw_response.jsonl'), lines=True, orient='records')

    # Also save metrics
    metrics = {k: v for k, v in results.items() if ('edited_pairs') not in k and ('raw_response') not in k}
    with open(os.path.join(args.output_dir, f'set_{args.dataset_name.replace("set_", "")}_{model_name.lower()}_{args.mode.lower()}_edit_metrics.jsonl'), 'w') as f:
        json.dump(metrics, f, indent=4)






if __name__ == "__main__":

    main()