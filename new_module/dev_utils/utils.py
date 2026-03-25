from typing import Union
import numpy as np
import pandas as pd
import yaml, torch

from new_module.set_consistency_energy.energynets.energynet import energynet

def read_outputs(file_path):
    outputs = pd.read_json(file_path, lines=True)
    outputs = outputs.explode('generations',ignore_index=True)
    outputs['prompt']=outputs['prompt'].apply(lambda x: x['text'])
    
    gen_keys = [set(x.keys()) for x in outputs['generations'].values]
    gen_key = set()
    for gen_key_i in gen_keys:
        gen_key |= gen_key_i
    
    for col in gen_key:
        outputs[col] = outputs['generations'].apply(lambda x: x.get(col,None))
        
    outputs.drop(columns=['generations'],inplace=True)
    return outputs


def ravel(unraveled_df):

    gen_keys = list(set(unraveled_df.columns) - {'prompt'})
        
    unraveled_df['generations']= unraveled_df.apply(lambda x: [{key: x[key] for key in gen_keys}],axis=1)
    prompt_list = unraveled_df['prompt'].tolist()
    return_df = []

    for prompt in prompt_list:

        generations_list = unraveled_df.loc[unraveled_df['prompt'] == prompt, 'generations'].tolist()
        generations_list = sum(generations_list, [])
        return_df.append({'prompt': {'text': prompt}, 
                          'generations': generations_list})

    return_df = pd.DataFrame.from_dict(return_df)
    return return_df

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

def precision_score_fn(gold_array, pred_array):
    
    pred = set(pred_array)
    gold = set(gold_array)
    tp = pred & gold

    if (len(pred) == 0):
        return 1
    else:
        return len(tp) / len(pred)

def recall_score_fn(gold_array, pred_array):

    pred = set(pred_array)
    gold = set(gold_array)
    tp = pred & gold

    if len(gold) == 0:
        return 1
    else:
        return len(tp) / len(gold)

def f1_score_fn(gold_array, pred_array):

    pred = set(pred_array)
    gold = set(gold_array)
    
    tp = pred & gold
    fp = pred - gold
    fn = gold - pred
    
    if (len(tp) == 0) and (len(fn) == 0) and (len(fp) == 0):
        return 1.0
    else:
        return (2 * len(tp)) / (2 * len(tp) + len(fp) + len(fn))


        
def load_sc_energy_model(config_path, device):
    
    model_config = yaml.load(open(config_path), 
                                Loader=yaml.FullLoader)
    model_config['device'] = device

    energy_net = energynet(params=model_config)
    energy_net.load_state_dict(torch.load(model_config["model_path"], 
                                        map_location=model_config['device'],
                                        weights_only=True)['state_dict'], strict=False)
    if 'threshold' in torch.load(model_config["model_path"],
                                map_location=model_config['device'],
                                weights_only=True):
        energy_net.threshold = torch.load(model_config["model_path"],
                                map_location=model_config['device'],
                                weights_only=True)['threshold']
    energy_net.eval()
    energy_net.to(device)
    
    return energy_net
    
    