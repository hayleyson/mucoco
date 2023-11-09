import pandas as pd

dat = pd.read_json('/home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/mlm-token-nps4-k3-weighted_sum-0.5-0.5/outputs_epsilon-3.txt',
                   lines=True)

original_dat = pd.read_json('/home/s3/hyeryung/mucoco/ell-e/toxicity-avoidance/data/testset_gpt2_2500.jsonl',
                            lines=True)

# dat.prompt = dat.prompt.applytext

untangled_dat = {'prompt': [], 'text': [], 'allsat': [], 'losses': [], 'weighted_loss': []}
for ix, row in dat.iterrows():
    
    prompt = row.prompt['text']
    corresponding_row = original_dat.loc[ix, :].copy()
    assert prompt == corresponding_row.prompt['text']
    for jx, gen in enumerate(row.generations):
        # first append original text
        untangled_dat['prompt'].append(prompt)
        untangled_dat['text'].append(corresponding_row.generations[jx]['text'])
        untangled_dat['losses'].append('original')
        untangled_dat['weighted_loss'].append('original')
        untangled_dat['allsat'].append('original')
    
        untangled_dat['prompt'].append(prompt)
        untangled_dat['text'].append(gen['text'])
        untangled_dat['losses'].append(gen['losses'])
        untangled_dat['weighted_loss'].append(gen['weighted_loss'])
        untangled_dat['allsat'].append(gen['allsat'])
    
final_dat = pd.DataFrame(untangled_dat)
final_dat.to_excel('/home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/mlm-token-nps4-k3-weighted_sum-0.5-0.5/outputs_epsilon-3.xlsx')        
    