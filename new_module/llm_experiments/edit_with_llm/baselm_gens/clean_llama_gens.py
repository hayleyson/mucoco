import json
import os
from transformers import AutoTokenizer


gendir = '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens'
genfiles = os.listdir(gendir)

for fpath_ in genfiles:
    
    if fpath_.startswith('llama2') and fpath_.endswith('gens.jsonl'):
        
        print("cleaning ...", fpath_)
        
        fpath = os.path.join(gendir,fpath_)
        fname, fext = os.path.splitext(fpath)

        with open(fpath, 'r') as f:
            data = f.readlines()    
        data = [json.loads(line) for line in data]


        for i, p_gen in enumerate(data): 
            
            data[i]['generations'] = [{'text': ' ' + x['text'].replace('</s>', '').replace('<unk>', '').replace('<s>', '')} if x['text'][0].isalpha() else {'text': x['text'].replace('</s>', '').replace('<unk>', '').replace('<s>', '')} for x in p_gen['generations']]

        with open(fname+"_cleaned"+fext, 'w') as f:
        # with open(fname+fext, 'w') as f:
            f.writelines([json.dumps(s) + '\n' for s in data])    



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
gendir = '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens'
genfiles = os.listdir(gendir)

for fpath_ in genfiles:
    
    if fpath_.startswith('llama3') and fpath_.endswith('gens.jsonl'):
        
        print("cleaning ...", fpath_)
        
        fpath = os.path.join(gendir,fpath_)
        fname, fext = os.path.splitext(fpath)

        with open(fpath, 'r') as f:
            data = f.readlines()    
        data = [json.loads(line) for line in data]


        for i, p_gen in enumerate(data): 
            
            data[i]['generations'] = [{'text': tokenizer.decode(tokenizer(x['text']).input_ids,skip_special_tokens=True)} for x in p_gen['generations']]

        with open(fname+"_cleaned"+fext, 'w') as f:
        # with open(fname+fext, 'w') as f:
            f.writelines([json.dumps(s) + '\n' for s in data])    
