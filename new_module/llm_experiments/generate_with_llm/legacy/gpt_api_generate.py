import os, random, pprint, json, argparse, openai, time
from openai import OpenAI
from new_module.llm_experiments.generate_with_llm.prompts import get_prompt
from new_module.data.nli_ifeval.ifeval_prompts import get_ifeval_prompt

random.seed(999)

def parse_api_outputs(outputs):
    outputs = [c.message.content for o in outputs for c in o.choices]
    return outputs 

def generate_and_save_result(args):
    
    
    client = OpenAI(api_key=args.openai_api_key)

    with open(args.input_file_path,'r') as f:
        raw_data = f.readlines()
    if args.input_file_path.endswith('jsonl'): ##toxic,senti,nli
        prompts = [json.loads(line)['prompt']['text'] for line in raw_data]
    else:## txt file ##formality transfer
        prompts = [line.rstrip() for line in raw_data]

    system_prompt, user_prompt = get_prompt(args)
    print(f"system_prompt: {system_prompt}")
    print(f"user_prompt: {user_prompt}")
    
    ## generate responses
    responses = []
    f = open(args.file_save_path, 'w')
    max_prompt_count = len(prompts) if args.num_test_prompts == -1 else args.num_test_prompts
    start_time = time.time()
    for p in prompts[:max_prompt_count]:
        if args.prompt_type == "nli_ifeval":
            result = get_ifeval_prompt()
            user_content = user_prompt.format(prompt=p) + '\n' +result['instruction']
            constraint_ids = result['instruction_id_list']
            # print(f"user_content: {user_content}")
        else:
            user_content = user_prompt.format(prompt=p)
        if system_prompt != "":
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        else:
            messages = [{"role": "user", "content": user_content}]
        response = client.chat.completions.create(
            model=args.model,
            top_p=0.96,
            max_completion_tokens=args.max_tokens, ## gpt2랑 똑같이 하려면, n=1로 하고 max_token을 매번 sampling해서 call 해야 함. 그렇게 할지?
            n=args.num_return_sequences,
            # logprobs=True,
            # top_logprobs=10,
            messages=messages
        )
        responses.append(response)
        # break
        if args.prompt_type == "nli_ifeval":
            formatted_generated_text = {'prompt': {'text': p, 'full_prompt': user_content, 'instruction_id_list': constraint_ids},
                                    'generations': [{'text': x.message.content} for x in response.choices]}
        else:
            formatted_generated_text = {'prompt': {'text': p},
                                    'generations': [{'text': x.message.content} for x in response.choices]}
        
        f.write(json.dumps(formatted_generated_text) + '\n')
        f.flush() 
    f.close()
    end_time = time.time()
    print(f"Total time taken (seconds): {end_time - start_time}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--file_save_path', type=str)
    parser.add_argument('--input_file_path', type=str)
    parser.add_argument('--prompt_type', type=str)    
    parser.add_argument('--num_return_sequences', type=int, default=10)    
    parser.add_argument('--max_tokens', type=int, default=30)
    parser.add_argument('--num_test_prompts', type=int, default=-1)
    
    args = parser.parse_args()
    
    generate_and_save_result(args)
    
    
    