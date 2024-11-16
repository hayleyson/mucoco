import pandas as pd 
pd.set_option('display.max_columns', None)

result_path='/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_noprompt_150.jsonl-results.txt'

metrics = []
with open(result_path, 'r') as f:
    metrics = f.readlines()
    
# assumption: if multiple items for the same metric, latest one is the right one
metrics_dict = {}
for line in metrics:
    items = line.split(', ')
    for item in items:
        if '----' in item:
            continue
        if ': ' in item:
            key, val = item.split(': ')         
        else: 
            
            key, val = item.split(' ')
        metrics_dict[key.strip()] = [float(val)]
        
# print(metrics_dict)

metrics_df = pd.DataFrame(metrics_dict)
metrics_df[['avg_max_toxicity', 'toxic_probability_p', 'avg_toxicity', 'toxic_probability_s', 'ppl', 'total_ppl', 'fluent_proba', 'dist-3', 'repetition_rate']].to_csv(result_path.replace('.txt', '.csv'),index=False)
