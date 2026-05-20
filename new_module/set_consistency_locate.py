import os, json, torch, itertools, time, re, random, sys, pickle, argparse, yaml
from pathlib import Path

from tqdm import tqdm
import pandas as pd

from new_module.dev_utils.utils import precision_score_fn, recall_score_fn, f1_score_fn, load_sc_energy_model, _pkl_path, load_pickle_dataset, load_eval2_dataset
from new_module.locate.new_locate_utils import LocateMachine4SCE


sys.path.append("new_module/set_consistency_energy")
from baselines.LLM.lm_loader import lm_loader
from baselines.baseline_model import baseline_model
from energynets.decomposition.no_decomposition import no_decomposition_loader
from tasks.dataset_loader import concat_arbitrary_pairs
from trainer.modules import locate_baseline


def locate_ebm(locate_machine, data_loader, device, params):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_reasoning_tokens = 0
    total_tokens = 0
    
    data_len = 0
    pred_list = []
    gold_list = []

    for batch in data_loader:
        input_text = batch[0][0]
        gold = batch[0][1]
        gold = [g+1 for g in gold]
        pred = locate_machine.locate_multiple_instances_at_once(input_text)
        pred = [p+1 for p in pred]
        
        if set(pred) == set(gold):
            accuracy =1
        else:
            accuracy = 0

        # precision
        if len(pred) == 0:
            precision = 1
        else:
            precision = len(set(pred) & set(gold)) / len(set(pred))

        # recall
        if len(gold) == 0:
            recall = 1
        else:
            recall = len(set(pred) & set(gold)) / len(set(gold)) 

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall / (precision + recall)
        
        data_len += 1
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        pred_list.append(pred)
        gold_list.append(gold)
        
    if data_len == 0:
        acc = 1
        precision = 1
        recall = 1
        f1 = 1
    else:
        acc = total_accuracy / data_len
        precision = total_precision / data_len
        recall = total_recall / data_len
        f1 = total_f1 / data_len

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred': pred_list,
        'gold': gold_list,
    }

def locate_random(data_loader, device, params):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_reasoning_tokens = 0
    total_tokens = 0
    
    data_len = 0
    pred_list = []
    gold_list = []

    for batch in data_loader:
        input_text = batch[0][0]
        gold = batch[0][1]
        gold = [g+1 for g in gold]
        pred = random.sample(range(1, len(input_text)+1), len(gold))
        
        if set(pred) == set(gold):
            accuracy =1
        else:
            accuracy = 0

        # precision
        if len(pred) == 0:
            precision = 1
        else:
            precision = len(set(pred) & set(gold)) / len(set(pred))

        # recall
        if len(gold) == 0:
            recall = 1
        else:
            recall = len(set(pred) & set(gold)) / len(set(gold)) 

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall / (precision + recall)
        
        data_len += 1
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        pred_list.append(pred)
        gold_list.append(gold)
        
    if data_len == 0:
        acc = 1
        precision = 1
        recall = 1
        f1 = 1
    else:
        acc = total_accuracy / data_len
        precision = total_precision / data_len
        recall = total_recall / data_len
        f1 = total_f1 / data_len

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred': pred_list,
        'gold': gold_list,
    }

def main():

    parser = argparse.ArgumentParser('')
    parser.add_argument('model_id', type=str)
    parser.add_argument('--dataset_name', type=str, default='lconvqa', choices=['lconvqa', 'set_nli'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--reasoning_effort', type=str, default=None, choices=['none', 'low', 'medium', 'high'])
    parser.add_argument('--use_incon_samples', action='store_true')
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--ebm_config_path', type=str, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        device=device,
        batch_size=1,
    )

    random.seed(args.random_seed)

    # Create output directory if not existent.
    os.makedirs(args.output_dir, exist_ok=True)

    # test_data = pd.read_json(args.test_data_path, lines=True)
    # if args.use_only_incon:
    #     print('Evaluating only using inconsistent samples...')
    #     test_data = test_data.loc[test_data['generations'].apply(lambda x: x[0]['label']) == 'incon'].copy()
    # if args.sample_data:
    #     print(f'Sampling {args.n_samples} rows from the test data using random seed {args.random_seed}...')
    #     test_data = test_data.sample(args.n_samples, random_state=args.random_seed)

    # print(f"Number of rows for evaluation: {test_data.shape[0]}")

    if params['dataset_path'] is None:
        # Load C and I datasets where set size >= 4.
        dataset_name = params['dataset']
        # load_eval2_dataset uses random.sample when n_samples is not None; default -1 means "all".
        n_samples_kw = None if args.n_samples < 0 else args.n_samples
        test_datasets, test_steps_names = load_eval2_dataset(
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

    # Set model_name for output file name
    model_name = args.model_id.split('/')[-1]
    if args.reasoning_effort is not None:
        model_name += f"_{args.reasoning_effort}"

    # Initialize model
    if (args.model_id == 'ebm') or (args.model_id == 'classifier'):
        energynet = load_sc_energy_model(args.ebm_config_path, device)
        params_ebm = yaml.load(open(args.ebm_config_path, 'r'), Loader=yaml.FullLoader)
        params_ebm['device'] = device
        model = LocateMachine4SCE(params_ebm, energynet, 'lconvqa')
        test_steps = [no_decomposition_loader(e, tokenizer=energynet.representation_model.tokenizer, params=params_ebm).get_loader() for e in test_datasets]
    elif args.model_id == "random":
        model = None
        test_steps = [lm_loader(e, params=params).get_loader() for e in test_datasets]
    else:
        model = baseline_model(params, 'locate')
        test_steps = [lm_loader(e, params=params).get_loader() for e in test_datasets]


    # Run locate and save metrics/results
    results = {}
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_reasoning_tokens = 0
    total_completion_tokens = 0
    total_locate_time = 0
    for es_idx, data_loader in enumerate(test_steps):
        data_name = test_steps_names[es_idx]

        print(f"=========== {es_idx+1} - {data_name} ===========")
        print(f"dataset count: {len(data_loader)}")
        if (args.model_id == "ebm") or (args.model_id == "classifier"):
            start = time.time()
            locate_result = locate_ebm(model, data_loader, device=params['device'], params=params)
            end = time.time()
            locate_time = end - start
        elif args.model_id == "random":
            start = time.time()
            locate_result = locate_random(data_loader, device=params['device'], params=params)
            end = time.time()
            locate_time = end - start
        else:
            start = time.time()
            locate_result = locate_baseline(model, data_loader, device=params['device'], params=params)
            end = time.time()
            locate_time = end - start

        results[f"locate_accuracy-{es_idx+1}-{data_name}"]  = locate_result.get('accuracy', 0)
        results[f"locate_precision-{es_idx+1}-{data_name}"] = locate_result.get('precision', 0)
        results[f"locate_recall-{es_idx+1}-{data_name}"]    = locate_result.get('recall', 0)
        results[f"locate_f1-{es_idx+1}-{data_name}"]        = locate_result.get('f1', 0)
        results[f"locate_gold-{es_idx+1}-{data_name}"]      = locate_result.get('gold', [])
        results[f"locate_pred-{es_idx+1}-{data_name}"]      = locate_result.get('pred', [])
        results[f"locate_raw_response-{es_idx+1}-{data_name}"]      = locate_result.get('raw_response', [])
        results[f"locate_total_reasoning_tokens-{es_idx+1}-{data_name}"]      = locate_result.get('total_reasoning_tokens', 0)
        results[f"locate_total_completion_tokens-{es_idx+1}-{data_name}"]      = locate_result.get('total_completion_tokens', 0)
        results[f"locate_time_seconds-{es_idx+1}-{data_name}"]      = locate_time
        print(f"[{es_idx+1}] {data_name}\n\taccuracy:{locate_result['accuracy']:.3f}, precision:{locate_result['precision']:.3f}, recall:{locate_result['recall']:.3f}, f1:{locate_result['f1']:.3f}")

        total_accuracy += locate_result.get('accuracy', 0)
        total_precision += locate_result.get('precision', 0)
        total_recall += locate_result.get('recall', 0)
        total_f1 += locate_result.get('f1', 0)
        total_reasoning_tokens += locate_result.get('total_reasoning_tokens', 0)
        total_completion_tokens += locate_result.get('total_completion_tokens', 0)
        total_locate_time += locate_time

    results = {r: results[r] for r in sorted(results)}

    results['total_accuracy'] = total_accuracy / len(test_steps)
    results['total_precision'] = total_precision / len(test_steps)
    results['total_recall'] = total_recall / len(test_steps)
    results['total_f1'] = total_f1 / len(test_steps)
    results['total_reasoning_tokens'] = total_reasoning_tokens
    results['total_completion_tokens'] = total_completion_tokens
    results['total_locate_time_seconds'] = total_locate_time

    print(f"total_accuracy:{results['total_accuracy']:.3f}, total_precision:{results['total_precision']:.3f}, total_recall:{results['total_recall']:.3f}, total_f1:{results['total_f1']:.3f}")
    print(f"total_reasoning_tokens:{results['total_reasoning_tokens']}, total_completion_tokens:{results['total_completion_tokens']}")
    print(f"total_locate_time_in_seconds:{results['total_locate_time_seconds']}")

    # Format predicted and gold inconsistent pair indices to save.
    response_list = []
    raw_response_list = []
    for es_idx, data_loader in enumerate(test_steps):
        data_name = test_steps_names[es_idx]
        response_list.append({
            'data_name': data_name,
            'gold': results[f"locate_gold-{es_idx+1}-{data_name}"],
            'pred': results[f"locate_pred-{es_idx+1}-{data_name}"],
        })
        raw_response_list.append({
            'data_name': data_name,
            'raw_response': results[f"locate_raw_response-{es_idx+1}-{data_name}"],
        })

    _ds_stem = args.dataset_name.replace("set_", "")
    locate_label_comparison = pd.DataFrame.from_dict(response_list)
    locate_label_comparison.to_json(os.path.join(args.output_dir, f'set_{_ds_stem}_{model_name.lower()}_locate_result.jsonl'), lines=True, orient='records')
    raw_response_list = pd.DataFrame.from_dict(raw_response_list)
    raw_response_list.to_json(os.path.join(args.output_dir, f'set_{_ds_stem}_{model_name.lower()}_locate_raw_response.jsonl'), lines=True, orient='records')

    # Also save metrics
    metrics = {k: v for k, v in results.items() if ('gold' not in k) and ('pred') not in k and ('raw_response') not in k}
    with open(os.path.join(args.output_dir, f'set_{_ds_stem}_{model_name.lower()}_locate_metrics.jsonl'), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    main()