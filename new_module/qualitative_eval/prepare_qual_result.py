import pandas as pd

## pip install googletrans==3.1.0a0
# 출처: https://neos518.tistory.com/201 [As I've always been:티스토리]
from googletrans import Translator
from tqdm import tqdm
translator = Translator()

llama_result = pd.read_json('/data/hyeryung/mucoco/new_module/qualitative_eval/llama3_8b_editing_gpt2_gens_nontoxic_reformat.jsonl', lines=True)
le_result = pd.read_json('/data/hyeryung/mucoco/new_module/qualitative_eval/outputs_epsilon0.9.txt', lines=True)
orig_result = pd.read_json('/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500_locate_reformat.jsonl', lines=True)

orig_result['prompt'] = orig_result['prompt'].apply(lambda x: x['text'])
orig_result['generations'] = orig_result['generations'].apply(lambda x: x[0]['text'])

llama_result['prompt'] = llama_result['prompt'].apply(lambda x: x['text'])
llama_result['generations'] = llama_result['generations'].apply(lambda x: x[0]['text'])
llama_result = llama_result.rename(columns={'generations': 'A_generations'})

le_result['prompt'] = le_result['prompt'].apply(lambda x: x['text'])
le_result['generations'] = le_result['generations'].apply(lambda x: x[0]['text'])
le_result = le_result.rename(columns={'generations': 'B_generations'})


final_df = pd.DataFrame(columns=['#', 'prompt', 'name', 'generations', 'which is less toxic?', 'which is more fluent?'])

final_df_data = []
for i in tqdm(range(len(orig_result))):
    
    if orig_result['generations'][i].strip() == le_result['B_generations'][i].strip():
        continue
    else:
        final_df_data.append((i+1, orig_result['prompt'][i], 'original', orig_result['generations'][i], '', ''))
        try:
            final_df_data.append((i+1, '', 'original', translator.translate(orig_result['prompt'][i] + orig_result['generations'][i], dest='ko').text, '', ''))
        except:
            print(i, 'orig_translation_error')
            final_df_data.append((i+1, '', 'original', '', '', ''))
        final_df_data.append((i+1, '', 'A', llama_result['A_generations'][i], '', ''))
        try:
            final_df_data.append((i+1, '', 'A', translator.translate(orig_result['prompt'][i] + llama_result['A_generations'][i], dest="ko").text, '', ''))
        except:
            print(i, 'A_translation_error')
            final_df_data.append((i+1, '', 'A', '', '', ''))
        final_df_data.append((i+1, '', 'B', le_result['B_generations'][i], '', ''))
        try:
            final_df_data.append((i+1, '', 'B', translator.translate(orig_result['prompt'][i] + le_result['B_generations'][i], dest="ko").text, '', ''))
        except:
            print(i, 'B_translation_error')
            final_df_data.append((i+1, '', 'B', '', '', ''))
    
final_df = pd.DataFrame(final_df_data, columns=['#', 'prompt', 'name', 'generations', 'which is less toxic?', 'which is more fluent?'])

final_df.to_excel('/data/hyeryung/mucoco/new_module/qualitative_eval/qualitative_result_1.xlsx', index=False)