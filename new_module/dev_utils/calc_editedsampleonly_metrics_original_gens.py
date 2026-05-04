"""
이런 코드가 많이 필요한게 좋은건지 모르겠지만, 
특정 index의 sample만 가지고 결과를 다시 계산하는 코드
"""

import pandas as pd
import math
from glob import glob
import os
os.chdir('/home/hyeryung/data/mucoco')
from evaluation.prompted_sampling.evaluate import distinctness, repetition
from new_module.dev_utils.utils import read_metric_file
os.getcwd()
import argparse

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


if __name__ == "__main__":
    
    # Example arguments
    # index_files = 
    
    parser = argparse.ArgumentParser(description='A program to read raw metric files and calculate summary statistics only using a selected set of samples')
    parser.add_argument('--output_files', nargs='+', type=str, help='A list of output file paths. e.g. <path>/5_tox_loc_edit_38576.jsonl')
    parser.add_argument('--index_files', nargs='+', type=str, help='A list of index files')
    parser.add_argument('--nicknames', nargs='+', type=str, help='A nicknames to refer to each output file')
    parser.add_argument('--task', type=str, help='Name of task')
    parser.add_argument('--sbert', action='store_true', help='Whether to include bertscore in the summary statistics')
    args = parser.parse_args()


    # 24/12/18 수정: index가 이미 저장되어 있는 경우. "1 24 5 63 ..\n" 형태로 저장되어 있음.
    edited_ixs={}
    nicknames = args.nicknames
    # index_files = ['/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332_index.txt']
    # index_files = ['new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_to_0_9_index.txt']
    # index_files = ['/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_9_index.txt']
    index_files = args.index_files

    with open(index_files[0], 'r') as f:
        indexes = f.read()

    indexes = indexes.strip().split(' ')
    indexes = [int(x) for x in indexes]

    edited_ixs[nicknames[0]] = indexes


    # save only the rows corresponding to the edited_ixs in the original generations
    # example: 
    # output_files = ["/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned.jsonl", 
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned.jsonl",
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned.jsonl"]
    # output_files = ["/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl"]
    # output_files = ['saeheeeom/new_module/iter_loc_edit_qwen/edited/2_tox_edited_38592.jsonl_total_0']
    # output_files = ['saeheeeom/new_module/iter_loc_edit_qwen/final/5_tox_loc_edit_38576.jsonl']
    # output_files = ['new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl']
    output_files = args.output_files

    for i, suffix in enumerate(nicknames):
        print(suffix)

        output_file=[output_files[i]]
        outputs=pd.read_json(output_file[0], lines=True)
        outputs=unravel(outputs)
        # print(outputs)
        outputs=outputs.loc[edited_ixs[suffix]]
        
        outputs_reformat = reformat(outputs)
        
        print(outputs_reformat)
        # outputs_reformat.to_json('_'.join([os.path.splitext(output_file[0])[0], f"edited_by_{suffix}_unraveled.jsonl"]), lines=True, orient='records')

        outputs=ravel(outputs)
        # print(outputs)
        # outputs.to_json('_'.join([os.path.splitext(output_file[0])[0], f"edited_by_{suffix}.jsonl"]), lines=True, orient='records')



    ## raw metrics 파일에서 해당 index에 대한 metrics를 뽑아온다.
    # metrics=['fluency','ppl-big-qwen','repetitions','sentiment_ext','dist-3', 'sbert']
    # metrics=['fluency','ppl-big-qwen','repetitions','toxicity','toxicity_int', 'dist-3', 'sbert']
    # metrics=['fluency','ppl-big-qwen','repetitions','toxicity','toxicity_int', 'dist-3']
    if args.task == 'toxicity':
        metrics = ['fluency','ppl-big-qwen','repetitions','toxicity','toxicity_int', 'dist-3']
    elif args.task == 'sentiment':
        metrics = ['fluency','ppl-big-qwen','repetitions','sentiment_ext', 'dist-3']
    elif args.task == 'nli':
        metrics = ['fluency','ppl-big-qwen','repetitions','nli', 'dist-3']

    if args.sbert:
        metrics += ['sbert']

    # ## ppl-big
    # metric='ppl-big'
    # ppl_metrics=[]
    # total_ppl_metrics=[]

    # # result_files = ["/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned_results.txt.ppl-big", 
    # #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.ppl-big",
    # #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned_results.txt.ppl-big"]

    # result_files = [f"{output_files[0]}-results.txt.ppl-big"]


    # for i, suffix in enumerate(nicknames):
    #     print(suffix)

    #     result_file=[result_files[i]]
    #     if metric in ['repetitions','toxicity']:
    #         result=pd.read_json(result_file[0],lines=True)
    #     else:
    #         result=pd.read_csv(result_file[0],header=None)
        
    #     result=result.loc[edited_ixs[suffix]]
    #     metric_value=result[0].mean()
    #     ppl_metrics.append(metric_value)
    #     metric_value=math.exp(result[1].sum()/result[2].sum())
    #     total_ppl_metrics.append(metric_value)

    from glob import glob
    if len(glob(f"{output_files[0]}-results.txt.*")) > 0:
        result_file_prefix = f"{output_files[0]}-results.txt"
    elif len(glob(f"{output_files[0].replace('/outputs_', '/results_').replace('.txt', '-test.txt')}")) > 0:
        result_file_prefix = f"{output_files[0].replace('/outputs_', '/results_').replace('.txt', '-test.txt')}"

    print("result_file_prefix:", result_file_prefix)

    ## ppl-big
    metric='ppl-big-qwen'
    ppl_qwen_metrics=[]
    total_ppl_qwen_metrics=[]
    result_files = [result_file_prefix + ".ppl-big-qwen"]
    result_files_alt = []


    for i, suffix in enumerate(nicknames):
        print(suffix)

        result_file=[result_files[i]]
        if metric in ['repetitions','toxicity']:
            try:
                result=pd.read_json(result_file[0],lines=True, dtype=float)
            except:
                result=pd.read_json(result_files_alt[i],lines=True, dtype=float)
        else:
            try:
                result=pd.read_csv(result_file[0],header=None, dtype=float)
            except:
                result=pd.read_csv(result_files_alt[i],header=None, dtype=float)
        print(result.dtypes)
        result=result.loc[edited_ixs[suffix]]
        metric_value=result[0].mean()
        ppl_qwen_metrics.append(metric_value)
        print(type(result[1].sum()))
        print(type(result[2].sum()))
        metric_value=math.exp(float(result[1].sum())/float(result[2].sum()))
        total_ppl_qwen_metrics.append(metric_value)
        
    ## fluency
    metric='fluency'
    fluency_metrics=[]
    # result_files = ["/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned_results.txt.fluency", 
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.fluency",
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned_results.txt.fluency"]

    if 'saeheeeom' in output_files[0]:
        result_files = [result_file_prefix.replace('/final/', '/final_fluency/').replace('/edited/', '/edited_fluency/') + ".fluency"]
    else:
        result_files = [result_file_prefix + ".fluency"]

    for i, suffix in enumerate(nicknames):
        print(suffix)

        result_file=[result_files[i]]
        if metric in ['repetitions','toxicity']:
            result=pd.read_json(result_file[0],lines=True)
        else:
            result=pd.read_csv(result_file[0],header=None)
        
        result=result.loc[edited_ixs[suffix]]
        if 'LABEL_1' in result[0].unique():
            metric_value=result.loc[result[0]=='LABEL_1'].shape[0]/result.shape[0]
        else:
            metric_value=result.loc[result[0]==1].shape[0]/result.shape[0]
        fluency_metrics.append(metric_value)
        

    ## repetitions
    metric='repetitions'
    repetitions_metrics=[]
    # result_files = ["/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned_results.txt.repetitions", 
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.repetitions",
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned_results.txt.repetitions"]
    result_files = [result_file_prefix + ".repetitions"]


    for i, suffix in enumerate(nicknames):
        print(suffix)

        result_file=[result_files[i]]
        # print(result_file[0])
        if metric in ['repetitions','toxicity']:
            result=pd.read_json(result_file[0],lines=True)
        else:
            result=pd.read_csv(result_file[0],header=None)
        
        result=result.loc[edited_ixs[suffix]]
        if result.empty:
            metric_value = 0
        else:
            # print('!')
            metric_value = result.loc[~result['repeated_phrase'].isna(),:].shape[0]/result.shape[0]
        repetitions_metrics.append(metric_value)


    ## dist-3
    dist3_metrics= []
    # output_files = ["/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_cleaned.jsonl", 
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned.jsonl",
    #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_cleaned.jsonl"]
    output_files = output_files

    for i, suffix in enumerate(nicknames):
        print(suffix)

        output_file=[output_files[i]]
        outputs=pd.read_json(output_file[0], lines=True)
        outputs=unravel(outputs)
        # print(outputs)
        outputs=outputs.loc[edited_ixs[suffix]]
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
        avg_max_toxicity_metrics=[]
        # result_files = ["/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct_gens_results.txt.toxicity", 
        #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/llama2_13b_chat_gens_cleaned_results.txt.toxicity",
        #                 "/home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_gens_results.txt.toxicity"]
        result_files = [result_file_prefix + ".toxicity"]


        for i, suffix in enumerate(nicknames):
            print(suffix)

            result_file=[result_files[i]]
            if metric in ['repetitions','toxicity']:
                result=pd.read_json(result_file[0],lines=True)
            else:
                result=pd.read_csv(result_file[0],header=None)

            if result.empty:
                result_file=glob(f"outputs/toxicity/**/**/*{suffix}*/results_filled.txt.{metric}")
                if len(result_file) == 0:
                    result_file=glob(f"outputs/toxicity/**/*{suffix}*/results_filled.txt.{metric}")
                # print(result_file[0])
                if metric in ['repetitions','toxicity']:
                    result=pd.read_json(result_file[0],lines=True)
                else:
                    result=pd.read_csv(result_file[0],header=None)
            # print(result)
            result=unravel_toxicity_data(result)
            result=result.loc[edited_ixs[suffix]]
            avg_toxicity=result['toxicity'].mean()
            toxic_proba=result['toxicity'].apply(lambda x: 1 if x>0.5 else 0).mean()
            toxic_proba_75=result['toxicity'].apply(lambda x: 1 if x>0.75 else 0).mean()
            
            outputs=pd.read_json(output_file[0], lines=True)
            outputs=unravel(outputs)
            outputs=outputs.loc[edited_ixs[suffix]]
            outputs_result=pd.concat([outputs, result],axis=1)
            avg_max_toxicity=outputs_result.groupby('prompt')['toxicity'].max().mean()
            
            avg_max_toxicity_metrics.append(avg_max_toxicity)
            avg_toxicity_metrics.append(avg_toxicity)
            toxic_proba_metrics.append(toxic_proba)
            toxic_proba_75_metrics.append(toxic_proba_75)
        
    if 'sentiment_ext' in metrics:
        # ## sentiment_int
        # metric='sentiment_int'
        # result_files = [f"{output_files[0]}-results.txt.{metric}"]
        # positive_proba_int_metrics=[]
        # for i, suffix in enumerate(nicknames):
            
        #     result_file=result_files[i]
        #     result=pd.read_json(result_file,lines=True)
        #     result=result.loc[edited_ixs[suffix]]
        #     metric_value=result['label'].apply(lambda x: 1 if x == 'LABEL_1' else 0).mean()
        #     # avg_positivity=result['score'].mean()
        #     positive_proba_int_metrics.append(metric_value)
                
        ## sentiment_ext
        metric='sentiment_ext'
        result_files = [result_file_prefix +f".{metric}"]
        positive_proba_ext_metrics=[]
        for i, suffix in enumerate(nicknames):
            
            result_file=result_files[i]
            result=pd.read_json(result_file,lines=True)
            result=result.loc[edited_ixs[suffix]]
            metric_value=result['label'].apply(lambda x: 1 if x == 'POSITIVE' else 0).mean()
            # avg_positivity=result['score'].mean()
            positive_proba_ext_metrics.append(metric_value)

        # ## sentiment_ext
        # metric='sentiment_gpt4o'
        # result_files = [f"{output_files[0]}-results.txt.{metric}"]
        # positive_proba_gpt4o_metrics=[]
        # for i, suffix in enumerate(nicknames):
            
        #     result_file=result_files[i]
        #     result=pd.read_csv(result_file,header=None)
        #     result.columns=['label']
        #     result=result.loc[edited_ixs[suffix]]
        #     metric_value=result['label'].mean()
        #     # avg_positivity=result['score'].mean()
        #     positive_proba_gpt4o_metrics.append(metric_value)

    if 'nli' in metrics:
        
        metric='nli'
        result_files = [result_file_prefix +f".{metric}"]
        contra_proba_metrics=[]
        for i, suffix in enumerate(nicknames):
            
            result_file=result_files[i]
            result=read_metric_file(result_file,'nli')
            result=result.loc[edited_ixs[suffix]]
            metric_value=result['nli_class'].apply(lambda x: 1 if x == 'contradiction' else 0).mean()
            contra_proba_metrics.append(metric_value)

    ## sbertscore
    if 'sbert' in metrics:
        
        sbert_metrics=[]
        sbert_geq_5_counts=[]
        sbert_geq_5_ratios=[]
        for suffix in nicknames:
            
            result_files = [result_file_prefix + ".sbertscore"]
            with open(result_files[0] , 'r') as f:
                raw_data = f.readlines()
                tmp_data = []
                for x in raw_data[1:]:
                    try:
                        tmp_data.append(float(x.strip()))
                    except:
                        tmp_data.append(float("nan"))
                
            # print(outputs)
            result=pd.DataFrame({'sbert':tmp_data})
            result=result.loc[edited_ixs[suffix]]
            sbert_score = result.sbert.mean()
            sbert_count = result.loc[result.sbert>=0.5].shape[0]
            sbert_ratio = sbert_count / result.shape[0]
            sbert_metrics.append(sbert_score)        
            sbert_geq_5_counts.append(sbert_count)
            sbert_geq_5_ratios.append(sbert_ratio)


    ## putting all together
    # pd.DataFrame({'nicknames':["llama3_8b_instruct_gens","llama2_13b_chat_gens","gpt4o_gens"], 
    if args.task == 'toxicity':
        pd.DataFrame({'nicknames':[f"llm_gens_{nicknames[0]}"], 
                    'sbert': sbert_metrics if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    'sbert_count': sbert_geq_5_counts if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    'sbert_ratio': sbert_geq_5_ratios if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    'avg_max_toxicity':avg_max_toxicity_metrics,
                    'avg_toxicity':avg_toxicity_metrics,
                    'toxic_proba':toxic_proba_metrics,
                    'toxic_75_proba':toxic_proba_75_metrics,
                    'ppl_qwen':ppl_qwen_metrics,
                    'total_ppl_qwen':total_ppl_qwen_metrics,
                    'delta_ppl':['' for _ in range(len(nicknames))],
                    'fluency_metrics':fluency_metrics,
                    'dist-3':dist3_metrics,
                    'rep_rate':repetitions_metrics,
                    'num_edits': [len(edited_ixs[suffix]) for suffix in nicknames],
                }).to_csv(f"{output_files[0].split('.jsonl')[0]}-results_{nicknames[0]}.csv",index=False)
    elif args.task == 'sentiment':
        pd.DataFrame({'nicknames':[f"llm_gens_{nicknames[0]}"], 
                    'sbert': sbert_metrics if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    'sbert_count': sbert_geq_5_counts if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    'sbert_ratio': sbert_geq_5_ratios if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    # 'sentiment_int':positive_proba_int_metrics,
                    'sentiment_ext':positive_proba_ext_metrics,
                    # 'sentiment_gpt4o':positive_proba_gpt4o_metrics,
                    'ppl':ppl_qwen_metrics,
                    'total_ppl':total_ppl_qwen_metrics,
                    'delta_ppl':['' for _ in range(len(nicknames))],
                    'fluency_metrics':fluency_metrics,
                    'dist-3':dist3_metrics,
                    'rep_rate':repetitions_metrics,
                    'num_edits': [len(edited_ixs[suffix]) for suffix in nicknames],
                }).to_csv(f"{output_files[0].split('.jsonl')[0]}-results_{nicknames[0]}.csv",index=False)
        
    elif args.task == 'nli':
        pd.DataFrame({'nicknames':[f"llm_gens_{nicknames[0]}"], 
                    'sbert': sbert_metrics if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    'sbert_count': sbert_geq_5_counts if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    'sbert_ratio': sbert_geq_5_ratios if 'sbert' in metrics else ["" for _ in range(len(nicknames))],
                    # 'sentiment_int':positive_proba_int_metrics,
                    # 'sentiment_ext':positive_proba_ext_metrics,
                    # 'sentiment_gpt4o':positive_proba_gpt4o_metrics,
                    'contra_prob': contra_proba_metrics,
                    'ppl':ppl_qwen_metrics,
                    'total_ppl':total_ppl_qwen_metrics,
                    'delta_ppl':['' for _ in range(len(nicknames))],
                    'fluency_metrics':fluency_metrics,
                    'dist-3':dist3_metrics,
                    'rep_rate':repetitions_metrics,
                    'num_edits': [len(edited_ixs[suffix]) for suffix in nicknames],
                }).to_csv(f"{output_files[0].split('.jsonl')[0]}-results_{nicknames[0]}.csv",index=False)