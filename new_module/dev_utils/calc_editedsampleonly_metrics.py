import pandas as pd
from glob import glob
import os
os.chdir('/data/hyeryung/mucoco')
from evaluation.prompted_sampling.evaluate import distinctness, repetition
os.getcwd()


## func to read output file
def unravel(outputs_df):
    outputs_df=outputs_df.explode('generations',ignore_index=True)
    
    outputs_df['prompt']=outputs_df['prompt'].apply(lambda x: x['text'])
    
    outputs_df['text']=outputs_df['generations'].apply(lambda x: x['text'])
    
    gen_dict=outputs_df['generations'].values[0]
    
    for col in gen_dict.keys():
        outputs_df[col] = outputs_df['generations'].apply(lambda x: x.get(col,None))

    return outputs_df

def ravel(unraveled_df):
    if 'tokens' in unraveled_df:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text'],
                                                               'tokens': x['tokens']}],axis=1)
    else:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text']}],axis=1)
    return unraveled_df.groupby('prompt')['generations'].sum([]).reset_index()

def unravel_toxicity_data(df):
    df['toxicity']=df['allresponses'].apply(lambda x: [x[0]['attributeScores']['TOXICITY']['summaryScore']['value'] for x in list(x.values())])
    df=df.explode('toxicity',ignore_index=True)
    return df



# run_ids = ["qzu2dk28",
#     "leoiknh8",
# "898x95uo",
# "wgarjlit",
# "86vs4jys",
# "qqitrpld",
# "8qv0f6o3",
# "bx3qb540"]

run_ids = ["03inbk5m"]
## edited index를 뽑아오고
## get common indexs

outputs_dfs=[]
edited_ixs={}

for run_id in run_ids:

    output_file=[x for x in glob(f"outputs/toxicity/**/**/*{run_id}*/outputs_epsilon*.txt") if not x.endswith('filled.txt')]
    if len(output_file) == 0:
        output_file=[x for x in glob(f"outputs/toxicity/**/*{run_id}*/outputs_epsilon*.txt") if not x.endswith('filled.txt')]
    # print(output_file)
    outputs=pd.read_json(output_file[0], lines=True)
    outputs=unravel(outputs)[['prompt','text','edited']].copy()
    outputs_dfs.append(outputs)
    edited_ixs.update({run_id: sorted(list(set(outputs.loc[outputs['edited']].index.tolist())))})
    print(run_id, len(edited_ixs[run_id]))

## raw metrics 파일에서 해당 index에 대한 metrics를 뽑아온다.

metrics=['fluency','ppl-big','repetitions','toxicity','toxicity_int', 'dist-3', 'sbert']

## ppl-big
metric='ppl-big'
ppl_metrics=[]
for run_id in run_ids:

    result_file=glob(f"outputs/toxicity/**/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    if len(result_file) == 0:
        result_file=glob(f"outputs/toxicity/**/*{run_id}*/results_epsilon*-test.txt.{metric}")

    if metric in ['repetitions','toxicity']:
        result=pd.read_json(result_file[0],lines=True)
    else:
        result=pd.read_csv(result_file[0],header=None)
    
    result=result.loc[edited_ixs[run_id]]
    metric_value=result[0].mean()
    ppl_metrics.append(metric_value)
    
    
## fluency
metric='fluency'
fluency_metrics=[]
for run_id in run_ids:

    result_file=glob(f"outputs/toxicity/**/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    if len(result_file) == 0:
        result_file=glob(f"outputs/toxicity/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    # print(result_file[0])
    if metric in ['repetitions','toxicity']:
        result=pd.read_json(result_file[0],lines=True)
    else:
        result=pd.read_csv(result_file[0],header=None)
    
    result=result.loc[edited_ixs[run_id]]
    metric_value=result.loc[result[0]=='LABEL_1'].shape[0]/result.shape[0]
    fluency_metrics.append(metric_value)
    

## repetitions
metric='repetitions'
repetitions_metrics=[]
for run_id in run_ids:
    print(run_id)

    result_file=glob(f"outputs/toxicity/**/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    if len(result_file) == 0:
        result_file=glob(f"outputs/toxicity/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    # print(result_file[0])
    if metric in ['repetitions','toxicity']:
        result=pd.read_json(result_file[0],lines=True)
    else:
        result=pd.read_csv(result_file[0],header=None)
    
    result=result.loc[edited_ixs[run_id]]
    if result.empty:
        metric_value = 0
    else:
        # print('!')
        metric_value = result.loc[~result['repeated_phrase'].isna(),:].shape[0]/result.shape[0]
    repetitions_metrics.append(metric_value)

## toxicity
metric='toxicity'
avg_toxicity_metrics=[]
toxic_proba_metrics=[]
toxic_proba_75_metrics=[]
for run_id in run_ids:

    result_file=glob(f"outputs/toxicity/**/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    if len(result_file) == 0:
        result_file=glob(f"outputs/toxicity/**/*{run_id}*/results_epsilon*-test.txt.{metric}")

    if metric in ['repetitions','toxicity']:
        result=pd.read_json(result_file[0],lines=True)
    else:
        result=pd.read_csv(result_file[0],header=None)

    if result.empty:
        result_file=glob(f"outputs/toxicity/**/**/*{run_id}*/results_filled.txt.{metric}")
        if len(result_file) == 0:
            result_file=glob(f"outputs/toxicity/**/*{run_id}*/results_filled.txt.{metric}")
        # print(result_file[0])
        if metric in ['repetitions','toxicity']:
            result=pd.read_json(result_file[0],lines=True)
        else:
            result=pd.read_csv(result_file[0],header=None)
    # print(result)
    result=unravel_toxicity_data(result)
    result=result.loc[edited_ixs[run_id]]
    avg_toxicity=result['toxicity'].mean()
    toxic_proba=result['toxicity'].apply(lambda x: 1 if x>0.5 else 0).mean()
    toxic_proba_75=result['toxicity'].apply(lambda x: 1 if x>0.75 else 0).mean()
    avg_toxicity_metrics.append(avg_toxicity)
    toxic_proba_metrics.append(toxic_proba)
    toxic_proba_75_metrics.append(toxic_proba_75)
    
    
## toxicity_int
metric='toxicity_int'
avg_toxicity_metrics_int=[]
toxic_proba_metrics_int=[]
toxic_proba_75_metrics_int=[]
for run_id in run_ids:

    result_file=glob(f"outputs/toxicity/**/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    if len(result_file) == 0:
        result_file=glob(f"outputs/toxicity/**/*{run_id}*/results_epsilon*-test.txt.{metric}")
    # print(result_file[0])
    if metric in ['repetitions','toxicity']:
        result=pd.read_json(result_file[0],lines=True)
    else:
        result=pd.read_csv(result_file[0],header=None)
    
    result=result.loc[edited_ixs[run_id]]
    avg_toxicity=result[0].mean()
    toxic_proba=result[0].apply(lambda x: 1 if x>0.5 else 0).mean()
    toxic_proba_75=result[0].apply(lambda x: 1 if x>0.75 else 0).mean()
    avg_toxicity_metrics_int.append(avg_toxicity)
    toxic_proba_metrics_int.append(toxic_proba)
    toxic_proba_75_metrics_int.append(toxic_proba_75)
    
## dist-3

dist3_metrics=[]
for run_id in run_ids:
    # print(run_id)
    
    output_file=[x for x in glob(f"outputs/toxicity/**/**/*{run_id}*/outputs_epsilon*.txt") if not x.endswith('filled.txt')]
    if len(output_file) == 0:
        output_file=[x for x in glob(f"outputs/toxicity/**/*{run_id}*/outputs_epsilon*.txt") if not x.endswith('filled.txt')]
    # print(output_file)
    outputs=pd.read_json(output_file[0], lines=True)
    outputs=unravel(outputs)
    # print(outputs)
    outputs=outputs.loc[edited_ixs[run_id]]
    outputs=ravel(outputs)
    # print(outputs)
    _,_,dist3=distinctness(outputs)
    dist3_metrics.append(dist3)        

## sbertscore
sbert_metrics=[]
sbert_geq_5_counts=[]
sbert_geq_5_ratios=[]
for run_id in run_ids:
    
    output_file=[x for x in glob(f"outputs/toxicity/**/**/*{run_id}*/results*.txt.sbertscore") if not x.endswith('filled.txt')]
    if len(output_file) == 0:
        output_file=[x for x in glob(f"outputs/toxicity/**/*{run_id}*/results*.txt.sbertscore") if not x.endswith('filled.txt')]
    with open(output_file[0] , 'r') as f:
        raw_data = f.readlines()
        tmp_data = []
        for x in raw_data[1:]:
            try:
                tmp_data.append(float(x.strip()))
            except:
                tmp_data.append(float("nan"))
        
    # print(outputs)
    result=pd.DataFrame({'sbert':tmp_data})
    result=result.loc[edited_ixs[run_id]]
    sbert_score = result.sbert.mean()
    sbert_count = result.loc[result.sbert>=0.5].shape[0]
    sbert_ratio = sbert_count / result.shape[0]
    sbert_metrics.append(sbert_score)        
    sbert_geq_5_counts.append(sbert_count)
    sbert_geq_5_ratios.append(sbert_ratio)

## 새로운 확장자의 results 파일에 metrics를 쓴다.

## putting all together


pd.DataFrame({'run_ids':run_ids, 
             'sbert': sbert_metrics,
              'sbert_count': sbert_geq_5_counts,
              'sbert_ratio': sbert_geq_5_ratios,
              'avg_toxicity':avg_toxicity_metrics,
              'toxic_proba':toxic_proba_metrics,
              'toxic_75_proba':toxic_proba_75_metrics,
              'ppl':ppl_metrics,
              'delta_ppl':['' for _ in range(len(run_ids))],
              'fluency_metrics':fluency_metrics,
              'dist-3':dist3_metrics,
              'rep_rate':repetitions_metrics,
              'num_edits': [len(edited_ixs[run_id]) for run_id in run_ids],
        }).to_csv('/data/hyeryung/mucoco/new_module/dev_utils/edited_index_only_metrics_longform_500.csv',index=False)