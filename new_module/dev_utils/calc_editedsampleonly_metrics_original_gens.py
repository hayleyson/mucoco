import pandas as pd
import math
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

def reformat(unraveled_df):
    if 'tokens' in unraveled_df:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text'],
                                                               'tokens': x['tokens']}],axis=1)
    else:
        unraveled_df['generations']= unraveled_df.apply(lambda x: [{'text': x['text']}],axis=1)
    return_df = unraveled_df.copy()
    return_df['prompt'] = return_df['prompt'].apply(lambda x: {'text':x})
        
    return return_df

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



# run_ids = ["qzu2dk28",
# "wgarjlit",
# "8qv0f6o3",]

run_ids = ["r7kykwge",]
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


# import joblib 
# joblib.dump(edited_ixs["qzu2dk28"], "/data/hyeryung/mucoco/new_module/dev_utils/edited_ixes_qzu2dk28.pkl")

# save only the rows corresponding to the edited_ixs in the original generations
# output_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned.jsonl", 
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned.jsonl",
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned.jsonl"]
output_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl"]


for i, run_id in enumerate(run_ids):
    print(run_id)

    output_file=[output_files[i]]
    outputs=pd.read_json(output_file[0], lines=True)
    outputs=unravel(outputs)
    # print(outputs)
    outputs=outputs.loc[edited_ixs[run_id]]
    
    outputs_reformat = reformat(outputs)
    
    print(outputs_reformat)
    outputs_reformat.to_json('_'.join([os.path.splitext(output_file[0])[0], f"edited_by_{run_id}_unraveled.jsonl"]), lines=True, orient='records')

    outputs=ravel(outputs)
    # print(outputs)
    outputs.to_json('_'.join([os.path.splitext(output_file[0])[0], f"edited_by_{run_id}.jsonl"]), lines=True, orient='records')



## raw metrics 파일에서 해당 index에 대한 metrics를 뽑아온다.

metrics=['fluency','ppl-big','repetitions','toxicity','toxicity_int', 'dist-3', 'sbert']

## ppl-big
metric='ppl-big'
ppl_metrics=[]
total_ppl_metrics=[]

# result_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned_results.txt.ppl-big", 
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.ppl-big",
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned_results.txt.ppl-big"]

result_files = [f"{output_files[0]}-results.txt.ppl-big"]


for i, run_id in enumerate(run_ids):
    print(run_id)

    result_file=[result_files[i]]
    if metric in ['repetitions','toxicity']:
        result=pd.read_json(result_file[0],lines=True)
    else:
        result=pd.read_csv(result_file[0],header=None)
    
    result=result.loc[edited_ixs[run_id]]
    metric_value=result[0].mean()
    ppl_metrics.append(metric_value)
    metric_value=math.exp(result[1].sum()/result[2].sum())
    total_ppl_metrics.append(metric_value)
    
    
## fluency
metric='fluency'
fluency_metrics=[]
# result_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned_results.txt.fluency", 
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.fluency",
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned_results.txt.fluency"]

result_files = [f"{output_files[0]}-results.txt.fluency"]

for i, run_id in enumerate(run_ids):
    print(run_id)

    result_file=[result_files[i]]
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
# result_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned_results.txt.repetitions", 
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.repetitions",
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned_results.txt.repetitions"]
result_files = [f"{output_files[0]}-results.txt.repetitions"]


for i, run_id in enumerate(run_ids):
    print(run_id)

    result_file=[result_files[i]]
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
# result_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_results.txt.toxicity", 
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.toxicity",
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_results.txt.toxicity"]
result_files = [f"{output_files[0]}-results.txt.toxicity"]


for i, run_id in enumerate(run_ids):
    print(run_id)

    result_file=[result_files[i]]
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
    

## dist-3
dist3_metrics= []
# output_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned.jsonl", 
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned.jsonl",
#                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned.jsonl"]
output_files = output_files


for i, run_id in enumerate(run_ids):
    print(run_id)

    output_file=[output_files[i]]
    outputs=pd.read_json(output_file[0], lines=True)
    outputs=unravel(outputs)
    # print(outputs)
    outputs=outputs.loc[edited_ixs[run_id]]
    outputs=ravel(outputs)
    # print(outputs)
    _,_,dist3=distinctness(outputs)
    dist3_metrics.append(dist3)        

# ## sbertscore
# sbert_metrics=[]
# sbert_geq_5_counts=[]
# sbert_geq_5_ratios=[]
# for run_id in run_ids:
    
#     output_file=[x for x in glob(f"outputs/toxicity/**/**/*{run_id}*/results*.txt.sbertscore") if not x.endswith('filled.txt')]
#     if len(output_file) == 0:
#         output_file=[x for x in glob(f"outputs/toxicity/**/*{run_id}*/results*.txt.sbertscore") if not x.endswith('filled.txt')]
#     with open(output_file[0] , 'r') as f:
#         raw_data = f.readlines()
#         tmp_data = []
#         for x in raw_data[1:]:
#             try:
#                 tmp_data.append(float(x.strip()))
#             except:
#                 tmp_data.append(float("nan"))
        
#     # print(outputs)
#     result=pd.DataFrame({'sbert':tmp_data})
#     result=result.loc[edited_ixs[run_id]]
#     sbert_score = result.sbert.mean()
#     sbert_count = result.loc[result.sbert>=0.5].shape[0]
#     sbert_ratio = sbert_count / result.shape[0]
#     sbert_metrics.append(sbert_score)        
#     sbert_geq_5_counts.append(sbert_count)
#     sbert_geq_5_ratios.append(sbert_ratio)

## 새로운 확장자의 results 파일에 metrics를 쓴다.

## putting all together


# pd.DataFrame({'run_ids':["llama3_8b_instruct_gens","llama2_13b_chat_gens","gpt4o_gens"], 
pd.DataFrame({'run_ids':[f"llm_gens_{run_ids[0]}"], 
             'sbert': ["" for _ in range(len(run_ids))],
              'sbert_count': ["" for _ in range(len(run_ids))],
              'sbert_ratio': ["" for _ in range(len(run_ids))],
              'avg_toxicity':avg_toxicity_metrics,
              'toxic_proba':toxic_proba_metrics,
              'toxic_75_proba':toxic_proba_75_metrics,
              'ppl':ppl_metrics,
              'total_ppl':total_ppl_metrics,
              'delta_ppl':['' for _ in range(len(run_ids))],
              'fluency_metrics':fluency_metrics,
              'dist-3':dist3_metrics,
              'rep_rate':repetitions_metrics,
              'num_edits': [len(edited_ixs[run_id]) for run_id in run_ids],
        }).to_csv(f"{output_files[0]}-results_editedonly.csv",index=False)