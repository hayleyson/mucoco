import os, json, torch, itertools, time, re, random, sys, pickle, argparse
from pathlib import Path
from copy import deepcopy

from openai import OpenAI
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

from new_module.dev_utils.utils import precision_score_fn, recall_score_fn, f1_score_fn, load_sc_energy_model

sys.path.append("new_module/set_consistency_energy")
from baselines.LLM.lm_loader import lm_loader
from baselines.baseline_model import baseline_model
from tasks.dataset_loader import concat_arbitrary_pairs
from trainer.modules import locate_baseline
from energynets.decomposition.no_decomposition import no_decomposition_loader

# =========================
# set_consistency_dataset 로더 유틸
# =========================

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

class GPT():

    def __init__(self, model_id: str, reasoning_effort: str):
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        self.edit_prompt = """# Role
You are an expert logician.
# Task
Inspect the provided text and eliminate any logical contradiction by editing only a few of the question-answer pair(s).
# Requirements
- Edit only some of the question-answer pair(s).
- Resolve the contradiction in the text.
- Preserve the rest of the text unchanged.
- Do not change wording, order, punctuation, or capitalization outside the edited pair(s).
# Output Format
- Return only the fully revised text as plain text.
- Output exactly the revised text and nothing else.
- Do not include explanations or additional formatting.
# Final Check
Before finalizing, verify that the contradiction is resolved, only the edited pair(s) were changed, and the output is the complete revised text.
# Input
{input_text}
"""
        self.edit_with_locate_prompt = """# Role
You are an expert logician.
# Task
Inspect the provided text and eliminate any logical contradiction by editing only the specified question-answer pair(s).
# Requirements
- Edit only the indicated question-answer pair(s).
- Resolve the contradiction in the text.
- Preserve the rest of the text unchanged.
- Do not change wording, order, punctuation, or capitalization outside the edited pair(s).
# Output Format
- Return only the fully revised text as plain text.
- Output exactly the revised text and nothing else.
- Do not include explanations or additional formatting.
# Final Check
Before finalizing, verify that the contradiction is resolved, only the allowed pair index was edited, and the output is the complete revised text.
# Input
**Input Text:**
{input_text}
**Pair Indexes to Edit:**
- {locate_labels}"""

    def generate(self, prompt: str) -> tuple:

        if self.reasoning_effort is not None:
            response = self.client.chat.completions.create(
                    model=self.model_id, 
                    messages=[{
                    "role": "user",
                    "content": prompt  
                    }],
                    reasoning_effort=self.reasoning_effort
                )
        else:
            response = self.client.chat.completions.create(
                model=self.model_id, 
                messages=[{
                    "role": "user",
                    "content": prompt  
                    }],
            )

        output_text = response.choices[0].message.content
        total_generated_tokens = response.usage.completion_tokens
        
        # Extract reasoning tokens safely (defaults to 0 if the model doesn't use reasoning)
        reasoning_tokens = 0
        if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens

        return output_text, reasoning_tokens, total_generated_tokens

    def set_prompt(self, data: list, located_indexes: list= None):
        
        input_text = ""
        for j, pair in enumerate(data):
            input_text += f"({j+1}) {pair}"

        if located_indexes is not None:
            return self.edit_with_locate_prompt.format(input_text=input_text, locate_labels=located_indexes)
        else:
            return self.edit_prompt.format(input_text=input_text)

    def parse_response(self, response: str) -> list:
        
        # Parse the output text to extract the edited question-answer pairs.
        pairs = re.split(r'\((\d+)\)\s*', response)
        
        parsed_pairs = []
        for p in pairs:
            if p == '' or p.isdigit():
                continue
            try:
                q, a = p.split(',')
                parsed_pairs.append((q.replace('question: ', '').strip(), a.replace('answer: ', '').strip().rstrip('.'), None))
            except:
                print(f"Warning - failed to parse:\n{p}")
                parsed_pairs.append((p, None, None))
        
        return parsed_pairs
        
    def edit(self, data: list, located_indexes: list=None) -> str:
        
        prompt = self.set_prompt(data, located_indexes)
        print(f"prompt:\n {prompt}")
        response, r_tok, t_tok = self.generate(prompt)
        parsed_pairs = self.parse_response(response)

        return {"edited_pairs": parsed_pairs, 
                "raw_response": response, 
                "reasoning_tokens": r_tok, 
                "total_generated_tokens": t_tok}

class HFModel():

    def __init__(self, model_id: str, tensor_parallel_size: int=1):
        self.model_id = model_id
        
        self.model = LLM(
            model=self.model_id, 
            trust_remote_code=True, 
            tensor_parallel_size=tensor_parallel_size
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
            top_p=1e-10
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)


        self.edit_prompt = """# Role
You are an expert logician.
# Task
Inspect the provided text and eliminate any logical contradiction by editing only a few of the question-answer pair(s).
# Requirements
- Edit only some of the question-answer pair(s).
- Resolve the contradiction in the text.
- Preserve the rest of the text unchanged.
- Do not change wording, order, punctuation, or capitalization outside the edited pair(s).
# Output Format
- Return only the fully revised text as plain text.
- Output exactly the revised text and nothing else.
- Do not include explanations or additional formatting.
# Final Check
Before finalizing, verify that the contradiction is resolved, only the edited pair(s) were changed, and the output is the complete revised text.
# Input
{input_text}
"""
        self.edit_with_locate_prompt = """# Role
You are an expert logician.
# Task
Inspect the provided text and eliminate any logical contradiction by editing only the specified question-answer pair(s).
# Requirements
- Edit only the indicated question-answer pair(s).
- Resolve the contradiction in the text.
- Preserve the rest of the text unchanged.
- Do not change wording, order, punctuation, or capitalization outside the edited pair(s).
# Output Format
- Return only the fully revised text as plain text.
- Output exactly the revised text and nothing else.
- Do not include explanations or additional formatting.
# Final Check
Before finalizing, verify that the contradiction is resolved, only the allowed pair index was edited, and the output is the complete revised text.
# Input
**Input Text:**
{input_text}
**Pair Indexes to Edit:**
- {locate_labels}"""

    def generate_batch(self, prompts: list) -> tuple:
        """
        Runs batch generation for a list of prompts.
        Returns a tuple of (responses, reasoning_tokens_list, total_tokens_list).
        """
        # apply chat template
        print(f"Prompt example before applying chat template : {prompts[0]}")

        prompts = [[ {"role": "user", 
                      "content": p}] for p in prompts]
        prompts = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]

        print(f"Prompt example after applying chat template : {prompts[0]}")

        outputs = self.model.generate(prompts, self.sampling_params)
        
        responses = []
        reasoning_tokens_list = []
        total_tokens_list = []
        
        for o in outputs:
            out = o.outputs[0]
            responses.append(out.text.strip())
            
            # Total generated tokens
            total_tokens = len(out.token_ids)
            total_tokens_list.append(total_tokens)
            
            # Extract reasoning tokens safely (if the model/vLLM version supports it)
            reasoning_tokens = 0
            if hasattr(out, 'reasoning_token_ids') and out.reasoning_token_ids is not None:
                reasoning_tokens = len(out.reasoning_token_ids)
            reasoning_tokens_list.append(reasoning_tokens)

            print(f"Reasoning tokens: {reasoning_tokens}")
            print(f"Total generated tokens: {total_tokens}")
            
        return responses, reasoning_tokens_list, total_tokens_list

    def set_prompt(self, data: list, located_indexes: list=None):
        
        input_text = ""
        for j, pair in enumerate(data):
            input_text += f"({j+1}) {pair}"

        if located_indexes is not None:
            return self.edit_with_locate_prompt.format(input_text=input_text, locate_labels=located_indexes)
        else:
            return self.edit_prompt.format(input_text=input_text)

    def parse_response(self, response: str) -> list:
        
        # Parse the output text to extract the edited question-answer pairs.
        if '</think>' not in response:
            # The response got truncated before the reasoning completed.
            # In this case, we cannot extract the edited pairs.
            return []
        
        response = response.split('</think>')[-1].strip()
        pairs = re.split(r'\((\d+)\)\s*', response)
        
        parsed_pairs = []
        for p in pairs:
            if p == '' or p.isdigit():
                continue
            try:
                q, a = p.split(',')
                parsed_pairs.append((q.replace('question: ', '').strip(), a.replace('answer: ', '').strip().rstrip('.'), None))
            except:
                print(f"Warning - failed to parse:\n{p}")
                parsed_pairs.append((p, None, None))
        
        return parsed_pairs
        
    def edit(self, data_list: list, located_indexes_list: list=None) -> str:
        
        if located_indexes_list is not None:
            prompts = [self.set_prompt(data, located_indexes) for data, located_indexes in zip(data_list, located_indexes_list)]
        else:
            prompts = [self.set_prompt(data) for data in data_list]
        responses, r_toks, t_toks = self.generate_batch(prompts)
        parsed_pairs = [self.parse_response(response) for response in responses]

        return {"edited_pairs_list": parsed_pairs, 
                "raw_response_list": responses, 
                "total_reasoning_tokens": sum(r_toks), 
                "total_completion_tokens": sum(t_toks)}



def main():

    parser = argparse.ArgumentParser('')
    parser.add_argument('model_id', type=str)
    parser.add_argument('--output_dir', type=str, default=None, required=True)
    parser.add_argument('--config_path', type=str, default=None, required=True)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--reasoning_effort', type=str, default=None, choices=['none', 'low', 'medium', 'high'])
    parser.add_argument('--use_incon_samples', action='store_true')
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--mode', type=str, choices=['wo_locate', 'w_gt_locate', 'w_self_locate', 'w_ebm_locate', 'w_random_locate'], default='wo_locate')

    args = parser.parse_args()

    params = dict(
        dataset_path=args.dataset_path,
        dataset='lconvqa', # one of ['lconvqa', 'set_nli']
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
        # Load C and I datasets where set size >= 4.
        dataset_name = params['dataset']
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

        if args.use_incon_samples:
            
            print(f"Using samples from datasets: {test_steps_names}")
            test_datasets = [test_datasets[i] for i in range(len(test_datasets)) if ('incon' in test_steps_names[i])]
            test_steps_names = [test_steps_name for test_steps_name in test_steps_names if ('incon' in test_steps_name)]
            print(f"Using samples from inconsistent datasets: {test_steps_names}")

            test_samples = []
            for dataset in test_datasets:
                test_samples.extend(dataset.dataset)
            test_samples = random.sample(test_samples, args.n_samples)
            print(f"Number of samples selected: {len(test_samples)}")

            canonical_test_dataset = test_datasets[0]
            canonical_test_dataset.dataset = test_samples
            test_datasets = [canonical_test_dataset]
            test_steps_names = ['incon']

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
                    args.reasoning_effort)
    else:
        model = HFModel(args.model_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    energynet = load_sc_energy_model(args.config_path, device)

    located_indexes_dict = {}
    if args.mode == 'w_self_locate':
        with open(os.path.join(args.output_dir, f"set_lconvqa_{model_name.lower()}_locate_result.jsonl"), 'r') as f:
            for line in f:
                row = json.loads(line.strip())
                located_indexes_dict[row['data_name']] = row['pred']
    elif args.mode == 'w_ebm_locate':
        with open(os.path.join(args.output_dir, f"set_lconvqa_ebm_locate_result.jsonl"), 'r') as f:
            for line in f:
                row = json.loads(line.strip())
                located_indexes_dict[row['data_name']] = row['pred']
    elif args.mode == 'w_random_locate':
        with open(os.path.join(args.output_dir, f"set_lconvqa_random_locate_result.jsonl"), 'r') as f:
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
    parsed_results.to_json(os.path.join(args.output_dir, f'set_lconvqa_{model_name.lower()}_{args.mode.lower()}_edit_result.jsonl'), lines=True, orient='records')
    raw_responses = pd.DataFrame.from_dict(raw_response_list)
    raw_responses.to_json(os.path.join(args.output_dir, f'set_lconvqa_{model_name.lower()}_{args.mode.lower()}_edit_raw_response.jsonl'), lines=True, orient='records')

    # Also save metrics
    metrics = {k: v for k, v in results.items() if ('edited_pairs') not in k and ('raw_response') not in k}
    with open(os.path.join(args.output_dir, f'set_lconvqa_{model_name.lower()}_{args.mode.lower()}_edit_metrics.jsonl'), 'w') as f:
        json.dump(metrics, f, indent=4)






if __name__ == "__main__":

    main()