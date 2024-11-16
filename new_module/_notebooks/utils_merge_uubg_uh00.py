# import json 
    
# with open('/data/hyeryung/mucoco/outputs/toxicity/devset/uubgx657_uh00cjd4/uh00cjd4/outputs_epsilon0.9.txt', 'r') as f:
#     res_uh00 = f.readlines()    
# res_uh00_json = [json.loads(x) for x in res_uh00]

# with open('/data/hyeryung/mucoco/outputs/toxicity/devset/uubgx657_uh00cjd4/uubgx657/outputs_epsilon0.9.txt', 'r') as f:
#     res_uubg = f.readlines()    
# res_uubg = res_uubg[::-1]
# res_uubg_json  = [json.loads(x) for x in res_uubg]


# res_merged = res_uh00_json + res_uubg_json[-(250-len(res_uh00_json)):]

# res_merged = [json.dumps(x) for x in res_merged]

# with open('/data/hyeryung/mucoco/new_module/_notebooks/utils_merge_uubg_uh00.txt', 'w') as f:
#     f.writelines([x + '\n' for x in res_merged])

# import pandas as pd

# res_merged = pd.read_json('/data/hyeryung/mucoco/new_module/_notebooks/utils_merge_uubg_uh00.txt', lines=True)
# devset = pd.read_json('/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl', lines=True)

# res_merged['prompt_text'] = res_merged['prompt'].apply(lambda x: x['text'])
# devset['prompt_text'] = devset['prompt'].apply(lambda x: x['text']) 

# print((res_merged['prompt_text']== devset['prompt_text']).all())

# print(res_merged.loc[res_merged['prompt_text'] != devset['prompt_text'], ['prompt_text']])

# print(devset.loc[res_merged['prompt_text'] != devset['prompt_text'], ['prompt_text']])

import json
import pandas as pd

res_uh00 = pd.read_json('/data/hyeryung/mucoco/outputs/toxicity/devset/uubgx657_uh00cjd4/uh00cjd4/outputs_epsilon0.9.txt', lines=True)
res_uh00.columns = ['prompt', 'generations_uh00']
res_uh00['prompt'] = res_uh00['prompt'].apply(lambda x: x['text'])

res_uubg = pd.read_json('/data/hyeryung/mucoco/outputs/toxicity/devset/uubgx657_uh00cjd4/uubgx657/outputs_epsilon0.9.txt', lines=True)
res_uubg.columns = ['prompt', 'generations_uubg']
res_uubg['prompt'] = res_uubg['prompt'].apply(lambda x: x['text'])

devset = pd.read_json('/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl', lines=True)
devset['prompt_raw'] = devset['prompt']
devset['prompt'] = devset['prompt'].apply(lambda x: x['text'])

devset_merged = pd.merge(devset, res_uh00, on='prompt', how='left')
devset_merged = pd.merge(devset_merged, res_uubg, on='prompt', how='left')
devset_merged['prompt'] = devset_merged['prompt_raw']
del devset_merged['prompt_raw']

devset_merged['generations_merged'] = devset_merged['generations_uubg']
devset_merged.loc[devset_merged['generations_merged'].isna(), 'generations_merged'] = devset_merged.loc[devset_merged['generations_merged'].isna(), 'generations_uh00']
devset_merged.loc[devset_merged['generations_merged'].isna(), 'generations_merged'] = devset_merged.loc[devset_merged['generations_merged'].isna(), 'generations']

### 이부분 고치기 (generations)
devset_merged = devset_merged.loc[devset_merged['generations_merged'].apply(lambda x: len(x)) > 0, :].copy() 
devset_merged['generations'] = devset_merged['generations_merged']
del devset_merged['generations_merged']
del devset_merged['generations_uubg']
del devset_merged['generations_uh00']
devset_merged.to_json('/data/hyeryung/mucoco/outputs/toxicity/devset/uubgx657_uh00cjd4/outputs_epsilon0.9.txt', lines=True, orient='records')
devset_merged.to_excel('/data/hyeryung/mucoco/outputs/toxicity/devset/uubgx657_uh00cjd4/outputs_epsilon0.9.xlsx', index=False)