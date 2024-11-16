import json
import os
from transformers import AutoTokenizer


# gendir = '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens'
# genfiles = os.listdir(gendir)

# for fpath_ in genfiles:
    
#     if fpath_.startswith('llama2') and fpath_.endswith('gens.jsonl'):
        
#         print("cleaning ...", fpath_)
        
#         fpath = os.path.join(gendir,fpath_)
#         fname, fext = os.path.splitext(fpath)

#         with open(fpath, 'r') as f:
#             data = f.readlines()    
#         data = [json.loads(line) for line in data]


#         for i, p_gen in enumerate(data): 
            
#             data[i]['generations'] = [{'text': ' ' + x['text'].replace('</s>', '').replace('<unk>', '').replace('<s>', '')} if x['text'][0].isalpha() else {'text': x['text'].replace('</s>', '').replace('<unk>', '').replace('<s>', '')} for x in p_gen['generations']]

#         with open(fname+"_cleaned"+fext, 'w') as f:
#         # with open(fname+fext, 'w') as f:
#             f.writelines([json.dumps(s) + '\n' for s in data])    



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
gendir = '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens'
# genfiles = os.listdir(gendir)
# genfiles = ['/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/positive/llama3_8b_chat_prompting_gens_senti_0shot.jsonl',
#             '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/positive/llama3_8b_chat_prompting_gens_senti_3shot.jsonl']
# genfiles = ['/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/nontoxic/editing/llama3_8b_editing_gpt2_gens_nontoxic.jsonl']
genfiles = ['/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/nontoxic/editing/llama3_8b_editing_gpt2_gens_all_nontoxic.jsonl',
           '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/negative/editing/llama3_8b_editing_gpt2_gens_neg.jsonl',
           '/data/hyeryung/mucoco/new_module/llm_experiments/baselm_gens/llama3_8b_instruct/positive/editing/llama3_8b_editing_gpt2_gens_pos.jsonl']


# input_file_path = '/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl'
# with open(input_file_path,'r') as f:
#     raw_data = f.readlines()
# if input_file_path.endswith('jsonl'): ##toxic,senti
#     prompts = [json.loads(line)['prompt']['text'] for line in raw_data]
# else:## txt file ##formality transfer
#     prompts = [line.rstrip() for line in raw_data]

def cleanup_text(s):
    print(s)
    tmp_gen = s.split('\n\n')[0]
    tmp_gen = tmp_gen.split('\nExplanation:')[0]
    tmp_gen = tmp_gen.split('\nRationale:')[0]
    tmp_gen = tmp_gen.split('\nJustification:')[0]
    tmp_gen = tmp_gen.split('\nNote:')[0]
    tmp_gen = tmp_gen.split('\nThis edited continuation')[0]
    tmp_gen = tmp_gen.split('This edited continuation')[0]
    tmp_gen = tmp_gen.split('\nThe edited continuation')[0]
    tmp_gen = tmp_gen.split('\nJustified changes:')[0]
    tmp_gen = tmp_gen.split('\nOriginal:')[0]
    tmp_gen = tmp_gen.split('\nReasoning:')[0]
    tmp_gen = tmp_gen.split('\nSequence:')[0]
    tmp_gen = tmp_gen.split('\nFinal Sequence:')[0]
    tmp_gen = tmp_gen.split('(Note:')[0]
    tmp_gen = tmp_gen.split('\nYour response')[0]
    tmp_gen = tmp_gen.split('\n**Your response')[0]
    tmp_gen = tmp_gen.split('\n(Your response')[0]
    tmp_gen = tmp_gen.split('[Your response')[0]
    tmp_gen = tmp_gen.split('\nNext')[0]
    tmp_gen = tmp_gen.split('\nGenerated')[0]
    tmp_gen = tmp_gen.split('\nContinue the sequence:')[0]
    tmp_gen = tmp_gen.split('\nPrefix:')[0]
    tmp_gen = tmp_gen.split('\ncontinue writing:')[0]
    tmp_gen = tmp_gen.split('\nSequence:')[0]
    

    
    
    print(tmp_gen)
    return tmp_gen

for fpath_ in genfiles:
        
    print("cleaning ...", fpath_)
    
    fpath = os.path.join(gendir,fpath_)
    fname, fext = os.path.splitext(fpath)

    with open(fpath, 'r') as f:
        data = f.readlines()    
    data = [json.loads(line) for line in data]


    for i, p_gen in enumerate(data): 
        
        clean_text = [tokenizer.decode(tokenizer(x['text']).input_ids,skip_special_tokens=True) for x in p_gen['generations']]
        clean_text = [cleanup_text(x) for x in clean_text]
        data[i]['generations'] = [{'text': x} for x in clean_text]
        # data[i]['prompt'] = {'text': prompts[i]}

    with open(fname+"_cleaned"+fext, 'w') as f:
    # with open(fname+fext, 'w') as f:
        f.writelines([json.dumps(s) + '\n' for s in data])    
