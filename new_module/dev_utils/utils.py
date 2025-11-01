from typing import Union
import numpy as np
import pandas as pd

def read_outputs(file_path):
    outputs = pd.read_json(file_path, lines=True)
    outputs = outputs.explode('generations',ignore_index=True)
    # outputs['generations'] = outputs['generations'].apply(lambda x: [x])
    outputs['prompt']=outputs['prompt'].apply(lambda x: x['text'])
    
    # outputs['text']=outputs['generations'].apply(lambda x: x['text'])
    
    gen_keys = [set(x.keys()) for x in outputs['generations'].values]
    gen_key = set()
    for gen_key_i in gen_keys:
        gen_key |= gen_key_i
    
    for col in gen_key:
        outputs[col] = outputs['generations'].apply(lambda x: x.get(col,None))
        
    outputs.drop(columns=['generations'],inplace=True)
    return outputs


def ravel(unraveled_df):
    if 'tokens' in unraveled_df:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text'],
                                                               'tokens': x['tokens']}],axis=1)
    else:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text']}],axis=1)
    return_df = unraveled_df.groupby('prompt')['generations'].sum([]).reset_index()
    return_df['prompt'] = return_df['prompt'].apply(lambda x: {'text':x})
        
    return return_df


def unravel_toxicity_data(df):
    df['toxicity']=df['allresponses'].apply(lambda x: [x[0]['attributeScores']['TOXICITY']['summaryScore']['value'] for x in list(x.values())])
    df=df.explode('toxicity',ignore_index=True)
    return df

def read_metric_file(result_file, metric) -> Union[np.array, pd.DataFrame]:
        
    if metric == 'ppl-big-qwen':
        result=pd.read_csv(result_file,header=None)
        return result
    
    elif metric == 'fluency':
        result=pd.read_csv(result_file,header=None)
        if result.shape[1] == 2:
            result[0] = result[0].apply(lambda x: 1 if x=='LABEL_1' else 0)
            result.columns=['fluency_class','fluency_proba']
            return result
        else:
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
        return result['label'].apply(lambda x: 1 if x == 'POSITIVE' else 0).values
    
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