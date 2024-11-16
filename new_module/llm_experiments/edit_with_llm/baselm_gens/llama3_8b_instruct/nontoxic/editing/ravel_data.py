import pandas as pd
import json

def unravel(outputs_df):
    outputs_df=outputs_df.explode('generations',ignore_index=True)
    
    outputs_df['prompt']=outputs_df['prompt'].apply(lambda x: x['text'])
    
    outputs_df['text']=outputs_df['generations'].apply(lambda x: x['text'])
    
    gen_dict=outputs_df['generations'].values[0]
    
    for col in gen_dict.keys():
        outputs_df[col] = outputs_df['generations'].apply(lambda x: x.get(col,None))

    return outputs_df

def ravel(unraveled_df, prompts):
    if 'tokens' in unraveled_df:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text'],
                                                               'tokens': x['tokens']}],axis=1)
    else:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text']}],axis=1)
        
    return_df = []
    for i, p in enumerate(prompts):
        
        tmp_df = unraveled_df[unraveled_df['prompt']==p]
        tmp_df = tmp_df.groupby('prompt')['generations'].sum([]).reset_index()
        tmp_df['prompt'] = tmp_df['prompt'].apply(lambda x: {'text': x})
        return_df.append(tmp_df)
        
    return_df = pd.concat(return_df,axis=0)
        
    # return_df = unraveled_df.groupby('prompt')['generations'].sum([]).reset_index()
    # return_df['prompt'] = return_df['prompt'].apply(lambda x: {'text': x})
    return return_df

file_save_path='/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/nontoxic/editing/llama3_8b_editing_gpt2_gens_all_nontoxic.jsonl'
data = pd.read_json(file_save_path, lines=True)


with open('new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl','r') as f:
    raw_data = f.readlines()
prompts_raw = [json.loads(line)['prompt']['text'] for line in raw_data]

data.to_json(file_save_path.replace('.jsonl','_bckp.jsonl'), orient='records', lines=True)
data_1 = unravel(data)
data_2 = ravel(data_1,prompts_raw)
data_2.to_json(file_save_path, orient='records', lines=True)