import os
import pprint
import json
import argparse
import openai
from openai import OpenAI

from new_module.llm_experiments.prompts import get_prompt


def generate_and_save_result(args):
    
    
    client = OpenAI(api_key=args.openai_api_key)

    with open(args.input_file_path,'r') as f:
        raw_data = f.readlines()
    if args.input_file_path.endswith('testset_gpt2_2500_locate.jsonl'):
        prompts = [json.loads(line)['prompt'] for line in raw_data]
        gens = [json.loads(line)['gen'] for line in raw_data]
    nontoxic_prompt = get_prompt(args)
    
    ## generate responses
    responses = []
    f = open(args.file_save_path, 'w')
    max_prompt_count = len(prompts) if args.num_test_prompts == -1 else args.num_test_prompts
    for p,g in zip(prompts[:max_prompt_count], gens[:max_prompt_count]):
        response = client.chat.completions.create(
            model="gpt-4o",
            top_p=0.96,
            max_tokens=args.max_tokens, ## gpt2랑 똑같이 하려면, n=1로 하고 max_token을 매번 sampling해서 call 해야 함. 그렇게 할지?
            n=args.num_return_sequences,
            # logprobs=True,
            # top_logprobs=10,
            messages=[
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": nontoxic_prompt % (p,g)}
            ]
        )
        responses.append(response)
        # break
        formatted_generated_text = {'prompt': {'text': p},
                                    'initial_generations': {'text': g},
                                    'generations': [{'text': x.message.content} for x in response.choices]}
        
        f.write(json.dumps(formatted_generated_text) + '\n')
        f.flush() 
    f.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--file_save_path', type=str)
    parser.add_argument('--input_file_path', type=str)
    parser.add_argument('--prompt_type', type=str)    
    parser.add_argument('--num_return_sequences', type=int, default=10)    
    parser.add_argument('--max_tokens', type=int, default=30)
    parser.add_argument('--num_test_prompts', type=int, default=-1)
    
    args = parser.parse_args()
    
    generate_and_save_result(args)
    
    
    