from typing import Union
import numpy as np
import pandas as pd
import yaml, torch, pickle, sys
from pathlib import Path

sys.path.append("new_module/set_consistency_energy")
from energynets.energynet import energynet
from tasks.dataset_loader import concat_arbitrary_pairs

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

    # assumption: prompts would be all empty if the dataset is not a prompt-continuation dataset.
    if all([x=="" for x in unraveled_df['prompt'].tolist()]):
        
        df_temp = unraveled_df.copy()
        df_temp['generations'] = df_temp.apply(lambda x: [{k: x[k] for k in gen_keys}], axis=1)
        df_temp['prompt'] = df_temp['prompt'].apply(lambda _: {"text": ""})
        return df_temp[['prompt', 'generations']]
    else:
        df_temp = unraveled_df.copy()
        df_temp['gen_dict'] = df_temp.apply(lambda x: {k: x[k] for k in gen_keys}, axis=1)
        result = df_temp.groupby('prompt', sort=False)['gen_dict'].apply(list).reset_index()
        result['prompt'] = result['prompt'].apply(lambda x: {'text': x})
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
    
def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    set_consistency_dataset/{dataset_name}/ 경로의 피클 파일 경로를 반환.
    파일명 규칙: {dataset_name}_{split}_{NAME}_dataset.pickle
      예) lconvqa_test_C_dataset.pickle, lconvqa_test_CI_dataset.pickle
    name 인자는 "test_C", "test_CI" 등 split 접두사를 포함한 문자열을 기대.
    """
    base = Path("new_module/data/convqa")
    fname = f"{dataset_name}_{split}_{name}_dataset.pickle"
    return base / fname

def load_pickle_dataset(dataset_name: str, split: str, name: str):
    path = _pkl_path(dataset_name, split, name)
    with open(path, "rb") as f:
        ds = pickle.load(f)
    return ds

def load_eval2_dataset(dataset_name):
    # 규칙에 따라 Eval2 데이터셋을 로딩한다.
    eval2_con_dataset_arbitrary_pairs = load_pickle_dataset(dataset_name, "eval2", "C")
    eval2_incon_dataset_arbitrary_pairs = load_pickle_dataset(dataset_name, "eval2", "I")
    eval2_con_dataset_arbitrary_pairs.dataset = [t for t in eval2_con_dataset_arbitrary_pairs.dataset if len(t) >=4]
    eval2_incon_dataset_arbitrary_pairs.dataset = [t for t in eval2_incon_dataset_arbitrary_pairs.dataset if len(t) >=4]
    
    concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs([eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs], concat_num=2)
    concat3_dataset, concat3_names, concat3_set_sizes = concat_arbitrary_pairs([eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs], concat_num=3)
    concat4_dataset, concat4_names, concat4_set_sizes = concat_arbitrary_pairs([eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs], concat_num=4)

    eval2_steps_names = ['con', 'incon'] + concat2_names+ concat3_names+ concat4_names
    eval2_datasets = [eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs
            ] + concat2_dataset + concat3_dataset + concat4_dataset

    # Classification accuracy와 별개로 locate accuracy만 보고 싶기 때문에, incon인 샘플만 취한다.
    eval2_datasets = [eval2_datasets[i] for i in range(len(eval2_datasets)) if ('incon' in eval2_steps_names[i])]
    eval2_steps_names = [eval2_steps_name for eval2_steps_name in eval2_steps_names if ('incon' in eval2_steps_name)]
    print(f"Using samples from inconsistent datasets: {eval2_steps_names}")
    
    # 편의상 데이터셋을 하나로 합친다.
    eval2_samples = []
    for dataset in eval2_datasets:
        eval2_samples.extend(dataset.dataset)
    print(f"Num samples: {len(eval2_samples)}")
    
    canonical_eval2_dataset = eval2_datasets[0]
    canonical_eval2_dataset.dataset = eval2_samples

    return canonical_eval2_dataset

def remove_prompt(row):
    if row['prompt'] in row['text']:
        return row['text'].replace(row['prompt'], '')
    else:
        return row['text']
    
def check_prompt(row):
    
    if row['prompt'] in row['text']:
        return 'exact_match'
    if ' '.join(row['prompt'].split(' ')[:3]) in row['text']:
        return 'partial_match'
    else:
        return 'no_match'
    
def postprocess_prompted_generations(output_path):
    
    outputs = read_outputs(output_path)
    
    print('Number of cases where the prompt is repeated:')
    print(outputs.apply(check_prompt, axis=1).value_counts().sort_index())
    
    outputs['text'] = outputs.apply(remove_prompt, axis=1)
    outputs = ravel(outputs)
    
    return outputs