import numpy as np
import sys

## load devset indices
with open('/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set_index.txt', 'r') as f:
    dev_set_index = [int(x.strip()) for x in f.readlines()]
    
## load full eval results
# run_id = '4kp4ti6s'
run_id = sys.argv[1]
outputs_path = f'/data/hyeryung/mucoco/outputs/toxicity/final/{run_id}/outputs_epsilon0.9.txt'
results_prefix = f'/data/hyeryung/mucoco/outputs/toxicity/final/{run_id}/results_epsilon0.9-test.txt'

res_fp = open(f'/data/hyeryung/mucoco/outputs/toxicity/final/{run_id}/results_epsilon0.9-test-devset.txt', 'w')

# fluency
with open(results_prefix + '.fluency', 'r') as f:
    fluency = [1 if x.strip() == 'LABEL_1' else 0 for x in f.readlines()]
fluency = np.array(fluency)[dev_set_index]
res_fp.write(f'fluency: {np.mean(fluency)}\n')

# ppl
with open(results_prefix + '.ppl-big', 'r') as f:
    ppl = [float(x.strip().split(',')[0]) for x in f.readlines()] # three values: 
ppl = np.array(ppl)[dev_set_index]
res_fp.write(f'ppl: {np.mean(ppl)}\n')

# repetitions
with open(results_prefix + '.repetitions', 'r') as f:
    repetition = [1 if x.strip() != '{}' else 0 for x in f.readlines()]
repetition = np.array(repetition)[dev_set_index]
res_fp.write(f'repetition: {np.mean(repetition)}\n')

# sbertscore
with open(results_prefix + '.sbertscore', 'r') as f:
    sbertscore = [float(x.strip()) for x in f.readlines()[1:]]
sbertscore = np.array(sbertscore)[dev_set_index]
res_fp.write(f'sbertscore: {np.mean(sbertscore)}\n')
    
# toxicity
import pandas as pd
toxicity = pd.read_json(results_prefix + '.toxicity', lines = True)

def unravel_toxicity_data(df):
    df['toxicity']=df['allresponses'].apply(lambda x: [x[0]['attributeScores']['TOXICITY']['summaryScore']['value'] for x in list(x.values())])
    df['toxicity']=df['allresponses'].apply(lambda x: [x[0]['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'] for x in list(x.values())])
    df=df.explode('toxicity',ignore_index=True)
    return df
toxicity = unravel_toxicity_data(toxicity)
toxicity = toxicity.iloc[dev_set_index,:]
res_fp.write(f"avg toxicity: {toxicity.loc[:, 'toxicity'].mean()}\n")
res_fp.write(f"toxic proba: {(toxicity.loc[:, 'toxicity'] > 0.5).mean()}\n")
    
# dist-3
from evaluation.prompted_sampling.evaluate import distinctness
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


outputs=pd.read_json(outputs_path, lines=True)
outputs=unravel(outputs)
outputs=outputs.loc[dev_set_index]
outputs=ravel(outputs)
_,_,dist3=distinctness(outputs)
res_fp.write(f'dist3: {dist3}\n')

res_fp.close()

