import json
import pandas as pd

data = pd.read_json('/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_editing_gpt2_gens_nontoxic.jsonl', lines=True)

data_out = []
for i, row in data.iterrows():
    
    
    tmp_gen = row['generations'][0]['text'].split('\n\n')[0]
    tmp_gen = tmp_gen.split('\nExplanation:')[0]
    tmp_gen = tmp_gen.split('\nRationale:')[0]
    tmp_gen = tmp_gen.split('\nJustification:')[0]
    tmp_gen = tmp_gen.split('\nNote:')[0]
    tmp_gen = tmp_gen.split('\nThis edited continuation')[0]
    tmp_gen = tmp_gen.split('This edited continuation')[0]
    tmp_gen = tmp_gen.split('\nThe edited continuation')[0]
    tmp_data_out = {'prompt': {'text': row['prompt']['text']},
                    'generations': [{'text': tmp_gen}]}
    data_out.append(json.dumps(tmp_data_out)+'\n')
    
with open('/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_editing_gpt2_gens_nontoxic_reformat.jsonl', 'w') as f:
    f.writelines(data_out)
