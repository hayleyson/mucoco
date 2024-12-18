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

# run_ids = ["r7kykwge",]
# ## edited index를 뽑아오고
# outputs_dfs=[]
# edited_ixs={}

# for run_id in run_ids:

#     output_file=[x for x in glob(f"outputs/toxicity/**/**/*{run_id}*/outputs_epsilon*.txt") if not x.endswith('filled.txt')]
#     if len(output_file) == 0:
#         output_file=[x for x in glob(f"outputs/toxicity/**/*{run_id}*/outputs_epsilon*.txt") if not x.endswith('filled.txt')]
#     # print(output_file)
#     outputs=pd.read_json(output_file[0], lines=True)
#     outputs=unravel(outputs)[['prompt','text','edited']].copy()
#     outputs_dfs.append(outputs)
#     edited_ixs.update({run_id: sorted(list(set(outputs.loc[outputs['edited']].index.tolist())))})
#     print(run_id, len(edited_ixs[run_id]))

# 24/12/18 수정: index가 이미 저장되어 있는 경우. "1 24 5 63 ..\n" 형태로 저장되어 있음.
edited_ixs={}
run_ids = ['_below_nontoxic_threshold_0_95_332']
output_file = ['/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332_index.txt']

with open(output_file[0], 'r') as f:
    indexes = f.read()

indexes = indexes.strip().split(' ')
indexes = [int(x) for x in indexes]

edited_ixs['_below_nontoxic_threshold_0_95_332'] = indexes


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
    # outputs_reformat.to_json('_'.join([os.path.splitext(output_file[0])[0], f"edited_by_{run_id}_unraveled.jsonl"]), lines=True, orient='records')

    outputs=ravel(outputs)
    # print(outputs)
    # outputs.to_json('_'.join([os.path.splitext(output_file[0])[0], f"edited_by_{run_id}.jsonl"]), lines=True, orient='records')



## raw metrics 파일에서 해당 index에 대한 metrics를 뽑아온다.
# metrics=['fluency','ppl-big-qwen','repetitions','sentiment_int','sentiment_ext', 'sentiment_gpt4o', 'dist-3', 'sbert']
metrics=['fluency','ppl-big-qwen','repetitions','toxicity','toxicity_int', 'dist-3', 'sbert']

# ## ppl-big
# metric='ppl-big'
# ppl_metrics=[]
# total_ppl_metrics=[]

# # result_files = ["/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned_results.txt.ppl-big", 
# #                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.ppl-big",
# #                 "/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned_results.txt.ppl-big"]

# result_files = [f"{output_files[0]}-results.txt.ppl-big"]


# for i, run_id in enumerate(run_ids):
#     print(run_id)

#     result_file=[result_files[i]]
#     if metric in ['repetitions','toxicity']:
#         result=pd.read_json(result_file[0],lines=True)
#     else:
#         result=pd.read_csv(result_file[0],header=None)
    
#     result=result.loc[edited_ixs[run_id]]
#     metric_value=result[0].mean()
#     ppl_metrics.append(metric_value)
#     metric_value=math.exp(result[1].sum()/result[2].sum())
#     total_ppl_metrics.append(metric_value)
    

## ppl-big
metric='ppl-big-qwen'
ppl_qwen_metrics=[]
total_ppl_qwen_metrics=[]
result_files = [f"{output_files[0]}-results.txt.ppl-big-qwen"]


for i, run_id in enumerate(run_ids):
    print(run_id)

    result_file=[result_files[i]]
    if metric in ['repetitions','toxicity']:
        result=pd.read_json(result_file[0],lines=True)
    else:
        result=pd.read_csv(result_file[0],header=None)
    
    result=result.loc[edited_ixs[run_id]]
    metric_value=result[0].mean()
    ppl_qwen_metrics.append(metric_value)
    metric_value=math.exp(result[1].sum()/result[2].sum())
    total_ppl_qwen_metrics.append(metric_value)
    
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

if 'toxicity' in metrics:
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
    
if 'sentiment_ext' in metrics:
    ## sentiment_int
    metric='sentiment_int'
    result_files = [f"{output_files[0]}-results.txt.{metric}"]
    positive_proba_int_metrics=[]
    for i, run_id in enumerate(run_ids):
        
        result_file=result_files[i]
        result=pd.read_json(result_file,lines=True)
        result=result.loc[edited_ixs[run_id]]
        metric_value=result['label'].apply(lambda x: 1 if x == 'LABEL_1' else 0).mean()
        # avg_positivity=result['score'].mean()
        positive_proba_int_metrics.append(metric_value)
            
    ## sentiment_ext
    metric='sentiment_ext'
    result_files = [f"{output_files[0]}-results.txt.{metric}"]
    positive_proba_ext_metrics=[]
    for i, run_id in enumerate(run_ids):
        
        result_file=result_files[i]
        result=pd.read_json(result_file,lines=True)
        result=result.loc[edited_ixs[run_id]]
        metric_value=result['label'].apply(lambda x: 1 if x == 'LABEL_1' else 0).mean()
        # avg_positivity=result['score'].mean()
        positive_proba_ext_metrics.append(metric_value)

    ## sentiment_ext
    metric='sentiment_gpt4o'
    result_files = [f"{output_files[0]}-results.txt.{metric}"]
    positive_proba_gpt4o_metrics=[]
    for i, run_id in enumerate(run_ids):
        
        result_file=result_files[i]
        result=pd.read_csv(result_file,header=None)
        result.columns=['label']
        result=result.loc[edited_ixs[run_id]]
        metric_value=result['label'].mean()
        # avg_positivity=result['score'].mean()
        positive_proba_gpt4o_metrics.append(metric_value)

## putting all together
# pd.DataFrame({'run_ids':["llama3_8b_instruct_gens","llama2_13b_chat_gens","gpt4o_gens"], 
if 'toxicity' in metrics:
    pd.DataFrame({'run_ids':[f"llm_gens_{run_ids[0]}"], 
                'sbert': ["" for _ in range(len(run_ids))],
                'sbert_count': ["" for _ in range(len(run_ids))],
                'sbert_ratio': ["" for _ in range(len(run_ids))],
                'avg_toxicity':avg_toxicity_metrics,
                'toxic_proba':toxic_proba_metrics,
                'toxic_75_proba':toxic_proba_75_metrics,
                'ppl_qwen':ppl_qwen_metrics,
                'total_ppl_qwen':total_ppl_qwen_metrics,
                'delta_ppl':['' for _ in range(len(run_ids))],
                'fluency_metrics':fluency_metrics,
                'dist-3':dist3_metrics,
                'rep_rate':repetitions_metrics,
                'num_edits': [len(edited_ixs[run_id]) for run_id in run_ids],
            }).to_csv(f"{output_files[0].split('.jsonl')[0]}-results_{run_ids[0]}.csv",index=False)
elif 'sentiment_ext' in metrics:
    pd.DataFrame({'run_ids':[f"llm_gens_{run_ids[0]}"], 
                'sbert': ["" for _ in range(len(run_ids))],
                'sbert_count': ["" for _ in range(len(run_ids))],
                'sbert_ratio': ["" for _ in range(len(run_ids))],
                'sentiment_int':positive_proba_int_metrics,
                'sentiment_ext':positive_proba_ext_metrics,
                'sentiment_gpt4o':positive_proba_gpt4o_metrics,
                'ppl':ppl_qwen_metrics,
                'total_ppl':total_ppl_qwen_metrics,
                'delta_ppl':['' for _ in range(len(run_ids))],
                'fluency_metrics':fluency_metrics,
                'dist-3':dist3_metrics,
                'rep_rate':repetitions_metrics,
                'num_edits': [len(edited_ixs[run_id]) for run_id in run_ids],
            }).to_csv(f"{output_files[0].split('.jsonl')[0]}-results_{run_ids[0]}.csv",index=False)