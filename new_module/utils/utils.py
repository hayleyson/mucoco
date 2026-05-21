from typing import Union
import numpy as np
import pandas as pd

def read_outputs(file_path):
    outputs = pd.read_json(file_path, lines=True)
    outputs = outputs.explode('generations',ignore_index=True)
    
    prompt_keys = [set(x.keys()) if isinstance(x, dict) else set() for x in outputs['prompt'].values]
    prompt_key = set()
    for prompt_key_i in prompt_keys:
        prompt_key |= prompt_key_i
    prompt_key -= {'text'}
    
    for col in prompt_key:
        outputs['_'.join(['prompt', col])] = outputs['prompt'].apply(lambda x: x.get(col, None))
    
    outputs['prompt']=outputs['prompt'].apply(lambda x: x['text'])
    
    gen_keys = [set(x.keys()) if isinstance(x, dict) else set() for x in outputs['generations'].values]
    gen_key = set()
    for gen_key_i in gen_keys:
        gen_key |= gen_key_i
    gen_key -= {'text'}
    
    for col in gen_key:
        outputs['_'.join(['gen', col])] = outputs['generations'].apply(
            lambda x, c=col: x.get(c, None) if isinstance(x, dict) else None
        )
    outputs['text'] = outputs['generations'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
        
    outputs.drop(columns=['generations'],inplace=True)
    return outputs


def ravel(unraveled_df):

    all_keys = list(set(unraveled_df.columns) - {'prompt', 'text'})
    prompt_keys = [k for k in all_keys if k.startswith('prompt_')]
    gen_keys = [k for k in all_keys if k.startswith('gen_')]

    # assumption: prompts would be all empty if the dataset is not a prompt-continuation dataset.
    if all([x=="" for x in unraveled_df['prompt'].tolist()]):
        
        df_temp = unraveled_df.copy()
        df_temp['generations'] = df_temp.apply(lambda x: [{"text": x['text']} | {k.replace('gen_', ''): x[k] for k in gen_keys}], axis=1)
        
        if len(prompt_keys) > 0:
            df_temp['prompt'] = df_temp.apply(lambda x: {"text": ""} | {k.replace('prompt_', ''): x[k] for k in prompt_keys}, axis=1)
        else:
            df_temp['prompt'] = df_temp['prompt'].apply(lambda _: {"text": ""})
        return df_temp[['prompt', 'generations']]
    else:
        df_temp = unraveled_df.copy()
        df_temp['gen_dict'] = df_temp.apply(lambda x: {"text": x['text']} | {k.replace('gen_', ''): x[k] for k in gen_keys}, axis=1)

        # 'prompt_full_prompt' exists if it is an output file from nli_ifeval task.
        # use it if available.
        group_cols = (
            ['prompt_full_prompt']
            if 'prompt_full_prompt' in df_temp.columns
            else ['prompt']
        )
        agg: dict = {'gen_dict': list}
        if 'prompt' not in group_cols:
            agg['prompt'] = 'first'
        for pk in prompt_keys:
            if pk not in group_cols:
                agg[pk] = 'first'
        result = df_temp.groupby(group_cols, sort=False).agg(agg).reset_index()

        if len(prompt_keys) > 0:
            result['prompt'] = result.apply(
                lambda x: {"text": x['prompt']}
                | {k.replace('prompt_', ''): x[k] for k in prompt_keys},
                axis=1,
            )
        else:
            result['prompt'] = result['prompt'].apply(lambda x: {"text": x})
        result = result.rename(columns={'gen_dict': 'generations'})
        return result[['prompt', 'generations']]

def unravel(outputs_df):
    outputs_df=outputs_df.explode('generations',ignore_index=True)
    outputs_df['prompt']=outputs_df['prompt'].apply(lambda x: x['text'])
    outputs_df['generations']=outputs_df['generations'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
    outputs_df = outputs_df.dropna().reset_index(drop=True)
    return outputs_df

def unravel_toxicity_data(df):
    df['toxicity']=df['allresponses'].apply(lambda x: [x[0]['attributeScores']['TOXICITY']['summaryScore']['value'] for x in list(x.values())])
    df=df.explode('toxicity',ignore_index=True)
    return df

def unravel_nli(outputs_df):
    outputs_df=outputs_df.explode('generations',ignore_index=True)
    outputs_df['prompt']=outputs_df['prompt'].apply(lambda x: x['premise'])
    outputs_df['source'] = outputs_df['prompt'].apply(lambda x: x['hypothesis'])
    outputs_df['generations']=outputs_df['generations'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
    outputs_df = outputs_df.dropna().reset_index(drop=True)
    return outputs_df

def read_metric_file(result_file, metric) -> Union[np.array, pd.DataFrame]:
        
    if metric == 'ppl-big-qwen':
        result=pd.read_csv(result_file,header=None)
        result.columns = ['ppl', 'nll', 'num_tokens']
        return result
    
    elif metric == 'fluency':
        result=pd.read_csv(result_file,header=None)
        
        if result.shape[1] == 2:
            if 'LABEL_1' in result[0]:
                fluent_label = 'LABEL_1'
            else:
                fluent_label = 1 # 2026/03/07: evaluation code has changed to save 1 for fluent and 0 for not fluent.
            result[0] = result[0].apply(lambda x: 1 if x==fluent_label else 0)
            result.columns=['fluency_class','fluency_proba']
            return result
        else:
            # 2026/03/07: this part of code is not affected by the evaluation code change mentioned above.
            result_class_only = result.loc[result[0].isin(['LABEL_0','LABEL_1'])].copy()
            if result.shape[0] == result_class_only.shape[0]*2:
                ## if above is true, then first half of data is fluency class while second half is fluency proba
                return pd.DataFrame({'fluency_class': result_class_only[0].apply(lambda x: 1 if x=='LABEL_1' else 0).tolist(),
                                     'fluency_proba': result.iloc[result_class_only.shape[0]:,0].tolist()})
            else:
                return pd.DataFrame({'fluency_class': result_class_only[0].apply(lambda x: 1 if x=='LABEL_1' else 0).tolist()})
    
    elif metric == 'repetitions':
        result=pd.read_json(result_file,lines=True)
        if result.empty:
            return np.zeros(result.shape[0])
        else:
            return (~result['repeated_phrase'].isna()).apply(lambda x: 1 if x==True else 0).values
        
    elif metric == 'toxicity':
        result=pd.read_json(result_file,lines=True)
        return unravel_toxicity_data(result)
    
    elif metric == 'sentiment_ext':
        result=pd.read_json(result_file,lines=True)
        if 'POSITIVE' in result[0]:
            sentiment_label = 'POSITIVE'
        else:
            sentiment_label = 1 # 2026/03/07: evaluation code has changed to save 1 for posivie and 0 for not positive.
        return result['label'].apply(lambda x: 1 if x == sentiment_label else 0).values
    
    elif metric == 'formality_ext':
        result = pd.read_csv(result_file,header=None)
        return result[0].values
    
    elif metric == 'set-consistency':
        result = pd.read_csv(result_file,header=None)
        return result[0].values
    
    elif metric == 'sbertscore':
        with open(result_file , 'r') as f:
            raw_data = f.readlines()
            tmp_data = []
            for x in raw_data[1:]:
                try:
                    tmp_data.append(float(x.strip()))
                except:
                    tmp_data.append(float("nan"))
        return np.array(tmp_data)    
    
    elif metric == 'nli':
        with open(result_file, 'r') as f:
            data = f.readlines()
            data = [eval(x.strip()) for x in data]
            data = pd.DataFrame.from_dict(data)
        return data
    
    else:
        raise ValueError(f"Unknown metric {metric}") 
    

def read_nli_result(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [eval(x.strip()) for x in data]
    data = pd.DataFrame.from_dict(data)
    return data

        