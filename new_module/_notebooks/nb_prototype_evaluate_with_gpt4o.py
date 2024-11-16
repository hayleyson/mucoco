from openai import OpenAI
import os
import json
import pandas as pd
import numpy as np

file_path = '/data/hyeryung/mucoco/outputs/toxicity/llm/ip4tt24k/outputs_epsilon0.9.txt'
# file_path = '/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/sentiment/gpt-3.5-turbo-0125_pplm_prompts_noprompt_150.jsonl'

api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

system_prompt = """\"Classify each of the following text samples as either Positive or Negative based on their sentiment. Do not include a Neutral class, and ensure each sample is distinctly categorized as either Positive or Negative. The number of examples is 20. Ensure you label every example provided.  Provide the output in JSON format as follows: {'results': ['Positive', 'Negative', ...]}.\"
Text Samples:
"""

generations_df = pd.read_json(file_path,lines=True)
responses_by_prompt = []
responses_unravel = []
print(f"Number of prompts: {len(generations_df)}")
for i in range(len(generations_df)):
    
    prompt = generations_df['prompt'][i]['text']
    generations = generations_df['generations'][i]
    
    full_text = [prompt + x['text'] for x in generations]
    print(f"Number of generations for {i}th prompt: {len(full_text)}")
    formatted_full_text = ""
    for text in full_text:
        formatted_full_text += "'" + text + "'" + ',\n\n'

    response = client.chat.completions.create(model='gpt-4o-2024-08-06', 
                                              temperature = 0, n = 1, max_tokens=200, #logprobs=True, 
                                              response_format={ 'type': "json_object" },
                                              messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": formatted_full_text}
    ])
    
    
    result = json.loads(response.choices[0].message.content)
    result = result['results']
    result = [1 if x == "Positive" else 0 for x in result]
    print(f"Number of predictions for {i}th prompt: {len(result)}")
    assert len(full_text) == len(result)

    responses_by_prompt.append(result)
    responses_unravel.extend(result)
    
    # responses.append(response.choices[0].message.content)
    
with open(file_path + '-results.txt.sentiment_gpt4o' ,'w') as f:
    
    f.writelines([str(x) + '\n' for x in responses_unravel])


with open(file_path + '-results.txt' ,'a') as f:
    
    f.write(f'\npositive_proba_gpt4o: {np.mean(responses_unravel)}, positive_proba_gpt4o_std: {np.std(responses_unravel)}\n')