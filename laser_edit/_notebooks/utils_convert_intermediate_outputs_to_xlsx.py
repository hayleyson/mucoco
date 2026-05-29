import pandas as pd

intermed = pd.read_json('/home/hyeryung/data/mucoco/outputs/sentiment/final/ri45lvy5/outputs_epsilon0.9.txt.intermediate',
                        lines=True)
intermed = intermed.explode('generations')
intermed['prompt'] = intermed['prompt'].apply(lambda x: x['text'])
# intermed = intermed.loc[intermed['prompt']=='The president of the country']

for i in range(10):
    intermed[f'iter{i}_original_sentence'] = intermed['generations'].apply(lambda x: x.get(f'iter{i}_original_sentence', None))
    intermed[f'iter{i}_masked_sentence'] = intermed['generations'].apply(lambda x: x.get(f'iter{i}_masked_sentence', None))
    intermed[f'iter{i}_best_text'] = intermed['generations'].apply(lambda x: x.get(f'iter{i}_best_text', None))
    intermed[f'iter{i}_update'] = intermed['generations'].apply(lambda x: x.get(f'iter{i}_update', None))

intermed.to_excel('/home/hyeryung/data/mucoco/outputs/sentiment/final/ri45lvy5/sentiment_pos_bv1_ab_ri45lvy5_outputs_epsilon0.9.txt.intermediate.xlsx', index=False)