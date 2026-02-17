import pandas as pd

def check_prompt_repetition_modification(data, verbose = True):
    
    prompt_repeated_count = 0
    prompt_modified_count = 0
    for i in range(len(data)):
        if verbose:
            print('=' * 30)
        prompt = data.iloc[i]['prompt']
        text = data.iloc[i]['text']
        if verbose:
            print(f"Prefix: {prompt}")
            print(f"Refined: {text}")
        repeated = prompt in text
        prompt_modified = (not repeated) and (' '.join(prompt.split()[:3]) in text)
        if repeated:
            if verbose:
                print(f"🛑 PROMPT REPEATED")
            prompt_repeated_count += 1
        if prompt_modified:
            if verbose:
                print(f"⚠️ PROMPT MODIFIED")
            prompt_modified_count += 1
    return prompt_repeated_count, prompt_modified_count

def clean_generation(prefix, generation):
    
    generation_c = generation.replace(prefix, '')
    generation_c = generation_c.replace('Prefix:', '')
    generation_c = generation_c.replace('prefix:', '')
    generation_c = generation_c.replace('Refined:', '')
    generation_c = generation_c.replace('refined:', '')
    generation_c = generation_c.replace('Refined Continuation:', '').strip()
    generation_c = generation_c.replace('Refined continuation:', '').strip()
    generation_c = generation_c.replace('refined continuation:', '').strip()
    generation_c = generation_c.replace('refined Continuation:', '').strip()
    
    return generation_c