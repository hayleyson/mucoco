import os
import pprint
import json
import argparse
import openai
from openai import OpenAI


def generate_and_save_result(args):
    
    client = OpenAI(api_key=args.openai_api_key)

    ## read prompts
    with open('/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl','r') as f:
            raw_data = f.readlines()
    prompts = [json.loads(line)['prompt']['text'] for line in raw_data]

    ## generate responses
    responses = []
    gens = []
    for p in prompts:
        response = client.chat.completions.create(
            model="gpt-4o",
            top_p=0.96,
            max_tokens=30, ## gpt2랑 똑같이 하려면, n=1로 하고 max_token을 매번 sampling해서 call 해야 함. 그렇게 할지?
            n=10,
            # logprobs=True,
            # top_logprobs=10,
            messages=[
            {"role": "system", "content": "Given a prompt, generate a continuation to the prompt."},
            {"role": "user", "content": p}
            ]
        )
        responses.append(response)
        # break
        gens.append([x.message.content for x in response.choices])
        
    ## save generations to a file
    formatted_generated_text_list = []
    for p, g in zip(prompts, gens):
            
            formatted_generated_text = {'prompt': {'text': p},
                                        'generations': [{'text': x} for x in g]}
            
            formatted_generated_text_list.append(json.dumps(formatted_generated_text))
            
    with open(args.file_save_path, 'w') as f:
        f.writelines([s + '\n' for s in formatted_generated_text_list])    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--file_save_path', type=str)
    
    args = parser.parse_args()
    
    generate_and_save_result(args)
    
    
    