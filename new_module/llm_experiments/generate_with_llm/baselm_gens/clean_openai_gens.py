import json
import os
from transformers import AutoTokenizer


gendir = '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens'
# genfiles = os.listdir(gendir)


# genfiles = ['/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o/gpt4o_prompting_gens_nontoxic_3shot.jsonl',
#             '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o/gpt4o_prompting_gens_nontoxic_0shot.jsonl']

genfiles = ['/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/gpt4o/nontoxic/gpt4o_gens.jsonl']
for fpath_ in genfiles:
    

    
    print("cleaning ...", fpath_)
    
    fpath = os.path.join(gendir,fpath_)
    print(fpath)
    fname, fext = os.path.splitext(fpath)

    with open(fpath, 'r') as f:
        data = f.readlines()    
    data = [json.loads(line) for line in data]


    for i, p_gen in enumerate(data): 
        
        data[i]['generations'] = [{'text': ' ' + x['text']} if x['text'][0].isalpha() else {'text': x['text']} for x in p_gen['generations']]

    with open(fname+"_cleaned"+fext, 'w') as f:
    # with open(fname+fext, 'w') as f:
        f.writelines([json.dumps(s) + '\n' for s in data])    