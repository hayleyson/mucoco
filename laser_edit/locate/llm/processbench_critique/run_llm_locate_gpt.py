import argparse
import numpy as np
import os
import json
from collections import Counter
from openai import OpenAI
from datasets import load_dataset
import re
from tqdm import tqdm
import random 

random.seed(42)

def read_prm800k_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = json.loads(line)
            if line['label']['finish_reason'] == 'give_up':
                continue
            if line['label']['finish_reason'] == 'solution':
                label = -1
            else:
                label = len(line['label']['steps']) - 1 
            data.append({'problem': line['question']['problem'],
                        'steps': line['question']['pre_generated_steps'],
                        'label': label})
    return data
    

def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

def prepare_input_boxed(template, input_d):
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        choices=['gsm8k', 'math', 'olympiadbench', 'omnimath', 'prm800k'])
    parser.add_argument('--model_name', type=str, required=True, help="OpenAI model name (e.g., gpt-4o)")
    parser.add_argument("--output_dir", type=str, default='laser_edit/processbench_critique/outputs')
    parser.add_argument('--use_voting', action='store_true')
    parser.add_argument('--voting_n', type=int, default=8)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    TEMPLATE = open('laser_edit/processbench_critique/critique_template.txt').read().strip()


    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    for config in args.configs:
        if not args.use_voting:
            output_dir = os.path.join(args.output_dir, args.model_name)
        else:
            output_dir = os.path.join(args.output_dir, f'{args.model_name}_voting')
        os.makedirs(output_dir, exist_ok=True)

        
        if config == 'prm800k':
            input_data = read_prm800k_jsonl("laser_edit/data/prm800k/phase2_test.jsonl")
        else:
            input_data = load_dataset('Qwen/ProcessBench', split=config)    

        random_indexes = random.sample(list(range(len(input_data))), 100)
        
        res_data = []
        # for i in tqdm(range(len(input_data))):
        for i in random_indexes:
            d = input_data[i].copy()
            messages = prepare_input_boxed(TEMPLATE, d)
            
            if not args.use_voting:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    # temperature=0.0,
                    max_completion_tokens=4096
                )
                generated_critique = response.choices[0].message.content
                pred = extract_answer(generated_critique)
                try:
                    pred = int(pred)
                except:
                    pred = None
            else:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    # temperature=1.0,
                    n=args.voting_n,
                    max_completion_tokens=4096
                )
                generated_critique = [choice.message.content for choice in response.choices]
                preds = [extract_answer(e) for e in generated_critique]
                preds = [e for e in preds if e is not None]
                if len(preds) == 0:
                    pred = None
                else:
                    pred = Counter(preds).most_common(1)[0][0]
                    try:
                        pred = int(pred)
                    except:
                        pred = None

            d['generated_critique'] = generated_critique
            d['prediction'] = pred
            d['match'] = (pred == d['label'])

            res_data.append(d)


        error_data = [e for e in res_data if e['label'] != -1]
        correct_data = [e for e in res_data if e['label'] == -1]

        with open(os.path.join(output_dir, f'{config}_error.jsonl'), 'w') as f:
            for e in error_data:
                f.write(json.dumps(e) + '\n')
        with open(os.path.join(output_dir, f'{config}_correct.jsonl'), 'w') as f:
            for e in correct_data:
                f.write(json.dumps(e) + '\n')
        
        acc1 = np.mean([e['match'] for e in error_data]) * 100
        acc2 = np.mean([e['match'] for e in correct_data]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')


if __name__ == '__main__':
    main()
