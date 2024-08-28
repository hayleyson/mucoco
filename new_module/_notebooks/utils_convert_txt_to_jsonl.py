import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompt_fpath', type=str, required=True)
parser.add_argument('--gen_fpath', type=str, required=True)
parser.add_argument('--expected_nlines', type=int, required=False)
args = parser.parse_args()

orig_fpath = args.prompt_fpath
# orig_fpath = '/data/hyeryung/mixmatch/data/detoxic/dev_set_prompts.jsonl'
    
if orig_fpath != "":
    prompts = pd.read_json(orig_fpath, lines=True)
else:
    prompts = pd.DataFrame({'prompt': ['' for _ in range(args.expected_nlines)]})

fpath = args.gen_fpath
# fpath = '/data/hyeryung/mixmatch/output_samples/detoxic/mask_disc_max_len_12_jigsaw_clsf_data_detoxic_em_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_100.0_date_24_04_2024_15_45_58/opt_samples.txt'
with open(fpath, 'r') as f:
    data = f.readlines()

print(len(prompts))
print(len(data))

j = 0
i = 0 
gens = []
while i < (len(prompts)-1) and j < len(data):
    
    prompt_text = prompts.iloc[i, 0]
    next_prompt_text = prompts.iloc[i + 1, 0]
    
    print(f"i: {i}, prompt_text: {prompt_text}, next_prompt_text: {next_prompt_text}")
    print(f"j: {j}, data[j]: {data[j]}")
    gen = data[j]
    
    j += 1
    while j < len(data):
        print(f"j: {j}, data[j]: {data[j]}, data[j].startswith(prompt_text): {data[j].startswith(prompt_text)}, data[j].startswith(next_prompt_text): {data[j].startswith(next_prompt_text)}")
        if data[j].startswith(next_prompt_text):
            gens.append(gen.rstrip()[len(prompt_text):])
            break
        else:
            gen += data[j]
            j += 1
    i += 1
    
gen = ''
while j < len(data):
    gen += data[j]
    j += 1
gens.append(gen.rstrip()[len(prompt_text):])

if ".txt" in fpath:
    w_fpath = fpath.replace('.txt', '.jsonl')
else:
    w_fpath = fpath + ".jsonl"
pd.DataFrame({'prompt': [{'text': x} for x in prompts['prompt']], 'generations': [[{'text': x}] for x in gens]}).to_json(w_fpath, orient='records', lines=True)
