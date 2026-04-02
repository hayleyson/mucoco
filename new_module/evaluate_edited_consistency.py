import os, json, torch, argparse, time, sys, re, random, pickle
from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np
import torch.nn.functional as F

# Add energy model path to sys.path
sys.path.append("new_module/set_consistency_energy")
from new_module.dev_utils.utils import load_sc_energy_model
from energynets.decomposition.no_decomposition import no_decomposition_loader
from baselines.LLM.lm_loader import lm_loader
from baselines.baseline_model import baseline_model
from tasks.dataset_loader import concat_arbitrary_pairs


def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    set_consistency_dataset/{dataset_name}/ 경로의 피클 파일 경로를 반환.
    파일명 규칙: {dataset_name}_{split}_{NAME}_dataset.pickle
      예) lconvqa_test_C_dataset.pickle, lconvqa_test_CI_dataset.pickle
    name 인자는 "test_C", "test_CI" 등 split 접두사를 포함한 문자열을 기대.
    """
    base = Path("new_module/data/convqa")
    fname = f"{dataset_name}_{split}_{name}_dataset.pickle"
    return base / fname

def load_pickle_dataset(dataset_name: str, split: str, name: str):
    path = _pkl_path(dataset_name, split, name)
    with open(path, "rb") as f:
        ds = pickle.load(f)
    return ds


class DatasetWrapper:
    def __init__(self, dataset_list, data_name):
        self.dataset = dataset_list
        self.data_name = [data_name for _ in range(len(dataset_list))]

def main():
    parser = argparse.ArgumentParser(description='Evaluate consistency of edited outputs.')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the energy model config.')
    parser.add_argument('--edit_result_path', type=str, default=None, help='Path to the edit result JSONL file.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation results.')
    parser.add_argument('--eval_model_id', type=str, default='gpt-5-mini', help='Model to use for evaluation: "ebm" or an LLM ID (e.g., "gpt-5-mini")')
    parser.add_argument('--eval_reasoning_effort', type=str, default=None, choices=['none', 'low', 'medium', 'high'])
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()

    if args.eval_model_id == 'ebm' and args.config_path is None:
        parser.error("--config_path is required when --eval_model_id is 'ebm'")

    if (args.output_dir is None) and (args.edit_result_path is not None):
        args.output_dir = os.path.dirname(args.edit_result_path)
    elif (args.output_dir is None) and (args.edit_result_path is None):
        args.output_dir = Path(__file__).resolve().parent
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load edit results
    if args.edit_result_path is None:

        assert "incon_300" in args.output_dir
        # Load C and I datasets where set size >= 4.
        dataset_name = 'lconvqa'
        test_con_dataset_arbitrary_pairs = load_pickle_dataset(dataset_name, "test", "C")
        test_incon_dataset_arbitrary_pairs = load_pickle_dataset(dataset_name, "test", "I")
        test_con_dataset_arbitrary_pairs.dataset = [t for t in test_con_dataset_arbitrary_pairs.dataset if len(t) >=4]
        test_incon_dataset_arbitrary_pairs.dataset = [t for t in test_incon_dataset_arbitrary_pairs.dataset if len(t) >=4]
        
        # Concat them to obtain concat2/3/4 datasets
        concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=2)
        concat3_dataset, concat3_names, concat3_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=3)
        concat4_dataset, concat4_names, concat4_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=4)

        test_steps_names = ['con', 'incon'] + concat2_names+ concat3_names+ concat4_names
        test_datasets = [test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs
                ] + concat2_dataset + concat3_dataset + concat4_dataset

        print(f"Using samples from datasets: {test_steps_names}")
        test_datasets = [test_datasets[i] for i in range(len(test_datasets)) if ('incon' in test_steps_names[i])]
        test_steps_names = [test_steps_name for test_steps_name in test_steps_names if ('incon' in test_steps_name)]
        print(f"Using samples from inconsistent datasets: {test_steps_names}")

        test_samples = []
        for dataset in test_datasets:
            test_samples.extend(dataset.dataset)
        test_samples = random.sample(test_samples, args.n_samples)
        print(f"Number of samples selected: {len(test_samples)}")
        
        edit_results = [{'data_name': 'incon', 
                        'pred': test_samples}]
        args.edit_result_path = "set_lconvqa_no_edit_result.jsonl"
    else:
        edit_results = []
        with open(args.edit_result_path, 'r') as f:
            for line in f:
                edit_results.append(json.loads(line.strip()))

    # Try to load existing metrics
    metrics_path = args.edit_result_path.replace('_edit_result.jsonl', '_edit_metrics.jsonl')
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(args.output_dir, os.path.basename(args.edit_result_path).replace('_edit_result.jsonl', '_edit_metrics.jsonl'))
    
    existing_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            existing_metrics = json.load(f)
        print(f"Loaded existing metrics from {metrics_path}")

    # Try to load existing raw metrics
    raw_metrics_path = args.edit_result_path.replace('_edit_result.jsonl', '_edit_raw_metrics.jsonl')
    if not os.path.exists(raw_metrics_path):
        raw_metrics_path = os.path.join(args.output_dir, os.path.basename(args.edit_result_path).replace('_edit_result.jsonl', '_edit_raw_metrics.jsonl'))
        
    existing_raw_metrics = {}
    if os.path.exists(raw_metrics_path):
        with open(raw_metrics_path, 'r') as f:
            existing_raw_metrics = json.load(f)
        print(f"Loaded existing raw metrics from {raw_metrics_path}")

    # Initialize Energy Model if required
    if args.eval_model_id == 'ebm':
        energynet = load_sc_energy_model(args.config_path, device)
    
    # Common params for loaders
    
    params = {
        'dataset': 'lconvqa',
        'device': device,
        'batch_size': 1,
        'eval': {'batch_size': 1},
        'baseline': {
            'type': 'llm', 
            'model': args.eval_model_id,
            'shot_num': 5,
            'locate_type': 'all_in_one',
            'prediction_type': 'all_in_one',
            'reasoning_effort': args.eval_reasoning_effort,
        },
    }

    # Initialize Baseline GPT-5 model if required
    if args.eval_model_id != 'ebm':
        gpt_evaluator = baseline_model(params, 'prediction')

    all_avg_sc_energies = []
    all_contradiction_probas = []
    all_contradiction_probas_gpt = []

    for es_idx, row in enumerate(edit_results):
        data_name = row['data_name']
        outputs = row['pred'] # list of list of [q, a, None]
        # temporary fix to get rid of "." at the end of answers
        new_outputs = []
        
        for example in outputs:
            new_example = []
            broken = False
            for q, a, label in example:
                if a is None: # skip examples with weirdly parsed outputs
                    broken = True
                    break
                else:
                    new_example.append([q, a.rstrip('.'), label])
            if not broken:
                new_outputs.append(new_example)
            else:
                new_outputs.append([])
        outputs = new_outputs
        
        print(f"Evaluating {data_name}...")
        
        # Wrap into dataset object
        wrapped_ds = DatasetWrapper(outputs, 'lconvqa')
        
        if args.eval_model_id == 'ebm':
            # 1. Energy Model Evaluation
            edited_dataloader = no_decomposition_loader(wrapped_ds, tokenizer=energynet.representation_model.tokenizer, params=params)
            loader = edited_dataloader.get_loader(split='eval')
            
            sc_scores = []
            sc_preds = []
            incon_counts = 0
            for batch in loader:
                with torch.no_grad():
                    text_input = [b[0] for b in batch]
                    
                    if text_input[0].strip() == energynet.representation_model.tokenizer.cls_token: # logic to find out cases where parsing failed.
                        sc_scores.extend([torch.nan] * len(text_input))
                        sc_preds.extend([torch.nan] * len(text_input))
                        continue
                    output = energynet.energy_model(text_input, pair_only=True)["predictions"]
                    
                    if hasattr(energynet, 'output_form'):
                        output_form = energynet.output_form
                    else: # fallback
                        output_form = 'real_num'
                    
                    if output_form == 'real_num':
                        probs = output.reshape(-1)
                    elif output_form == '2dim_vec':
                        probs = F.softmax(output, dim=-1)[:, -1]
                    
                    sc_scores.extend(probs.tolist())
                    threshold = getattr(energynet, 'threshold', 0.5)
                    # incon_counts += torch.sum(torch.where(probs <= threshold, 0, 1)).item()
                    sc_preds.extend(torch.where(probs <= threshold, 0, 1).tolist())
            
            avg_sc_energy = np.nanmean(sc_scores) if sc_scores else 0
            # contradiction_proba = incon_counts / len(outputs) if outputs else 0
            contradiction_proba = np.nanmean(sc_preds) if sc_preds else 0
            
            # Store in metrics
            existing_metrics[f"avg_sc_energy-{es_idx+1}-{data_name}"] = avg_sc_energy
            existing_metrics[f"contradiction_proba-{es_idx+1}-{data_name}"] = contradiction_proba
            existing_raw_metrics[f"raw_sc_scores-{es_idx+1}-{data_name}"] = sc_scores
            existing_raw_metrics[f"raw_sc_preds-{es_idx+1}-{data_name}"] = sc_preds
            
            all_avg_sc_energies.append(avg_sc_energy)
            all_contradiction_probas.append(contradiction_proba)
        else:
            # 2. GPT Evaluation
            lm_loader_obj = lm_loader(wrapped_ds, params=params)
            lm_dataloader = lm_loader_obj.get_loader()
            
            preds = []
            print(f"Running GPT evaluation for {data_name}...")
            for i, pairs in enumerate(lm_dataloader):
                print(f"pairs: {pairs}")
                if len(pairs[0][0]) == 0: # logic to find out cases where parsing failed.
                    print(f"Skipping evaluation for this example!!!")
                    preds.extend([torch.nan] * len(pairs))
                    continue
                try:
                    evaluate_result = gpt_evaluator.predict(pairs)
                    preds.extend(evaluate_result['pred'])
                except Exception as e:
                    print(f"!!! GPT evaluation failed for index {i} in {data_name}: {e}")
                    # Fallback for failed prediction
                    preds.extend([torch.nan] * len(pairs)) 

            contradiction_proba_gpt = np.nanmean(preds) if preds else 0
            
            # Store in metrics
            existing_metrics[f"contradiction_proba_{args.eval_model_id.lower()}-{es_idx+1}-{data_name}"] = contradiction_proba_gpt
            existing_raw_metrics[f"raw_contradiction_pred_{args.eval_model_id.lower()}-{es_idx+1}-{data_name}"] = preds
            
            all_contradiction_probas_gpt.append(contradiction_proba_gpt)

    # Global scores
    if all_avg_sc_energies: existing_metrics["avg_sc_energy"] = sum(all_avg_sc_energies) / len(all_avg_sc_energies)
    if all_contradiction_probas: existing_metrics["contradiction_proba"] = sum(all_contradiction_probas) / len(all_contradiction_probas)
    if all_contradiction_probas_gpt: existing_metrics[f"contradiction_proba_{args.eval_model_id.lower()}"] = sum(all_contradiction_probas_gpt) / len(all_contradiction_probas_gpt)

    # Save metrics
    output_metrics_path = os.path.join(args.output_dir, os.path.basename(metrics_path))
    with open(output_metrics_path, 'w') as f:
        json.dump(existing_metrics, f, indent=4)
        
    # Save raw metrics
    output_raw_metrics_path = os.path.join(args.output_dir, os.path.basename(raw_metrics_path))
    with open(output_raw_metrics_path, 'w') as f:
        json.dump(existing_raw_metrics, f, indent=4)
    
    print(f"Evaluation complete. Metrics saved to {output_metrics_path}")
    print(f"Raw metrics saved to {output_raw_metrics_path}")

if __name__ == "__main__":
    main()
