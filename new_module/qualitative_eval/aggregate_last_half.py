import pandas as pd


a = pd.read_excel('/data/hyeryung/mucoco/new_module/qualitative_eval/locate_edit_qualitative_eval_000_last_half_윤섭.xlsx', 
                  sheet_name='labelling',
                  index_col=None)
# print(a.head())
# print(a.columns)
a = a.iloc[:, :-1]
a.columns = ['#', 'prompt', 'name', 'generations', 'toxicity', 'fluency', 'contents_pres', 'comment']
# print(a.head())
# print(a.columns)
# print(a.toxicity)
# print(a.fluency)
# print(a.contents_pres)
# print(a.comment)

a = a[['#', 'toxicity','fluency','contents_pres', 'comment']].copy().dropna(subset=['toxicity'])

b = pd.read_excel('/data/hyeryung/mucoco/new_module/qualitative_eval/locate_edit_qualitative_eval_000_last_half_윤섭.xlsx', 
                  sheet_name='labelling',
                  index_col=None)
# print(a.head())
# print(a.columns)
b = b.iloc[:, :-1]
b.columns = ['#', 'prompt', 'name', 'generations', 'toxicity', 'fluency', 'contents_pres', 'comment']
# print(a.head())
# print(a.columns)
# print(a.toxicity)
# print(a.fluency)
# print(a.contents_pres)
# print(a.comment)

b = b[['#', 'toxicity','fluency','contents_pres', 'comment']].copy().dropna(subset=['toxicity'])


c = pd.read_excel('/data/hyeryung/mucoco/new_module/qualitative_eval/locate_edit_qualitative_eval_000_last_half_윤섭.xlsx', 
                  sheet_name='labelling',
                  index_col=None)
# print(a.head())
# print(a.columns)
c = c.iloc[:, :-1]
c.columns = ['#', 'prompt', 'name', 'generations', 'toxicity', 'fluency', 'contents_pres', 'comment']
# print(a.head())
# print(a.columns)
# print(a.toxicity)
# print(a.fluency)
# print(a.contents_pres)
# print(a.comment)

c = c[['#', 'toxicity','fluency','contents_pres', 'comment']].copy().dropna(subset=['toxicity'])


a.columns = ['#'] + [f'{col}_a' for col in a.columns[1:]]
b.columns = ['#'] + [f'{col}_b' for col in b.columns[1:]]
c.columns = ['#'] + [f'{col}_c' for col in c.columns[1:]]

ab = pd.merge(a, b, on='#', how='inner')
print(ab.shape, a.shape)

abc = pd.merge(ab, c, on='#', how='inner')
print(abc.shape, c.shape)


### majority vote 


abc['toxicity'] = abc[['toxicity_a', 'toxicity_b', 'toxicity_c']].mode(axis=1)[0]
# print(abc['toxicity'])
abc['fluency'] = abc[['fluency_a', 'fluency_b', 'fluency_c']].mode(axis=1)[0]
abc['contents_pres'] = abc[['contents_pres_a', 'contents_pres_b', 'contents_pres_c']].mode(axis=1)[0]


abc.to_excel('/data/hyeryung/mucoco/new_module/qualitative_eval/locate_edit_qualitative_eval_000_last_half_summary.xlsx', 
             sheet_name='raw',
             index=False)

with open('/data/hyeryung/mucoco/new_module/qualitative_eval/locate_edit_qualitative_eval_000_last_half_summary.txt', 'w') as f:
    print('-'*50, file=f)
    print(abc.toxicity.value_counts(normalize=True).sort_index(), file=f)
    print('-'*50, file=f)
    print(abc.fluency.value_counts(normalize=True).sort_index(), file=f)
    print('-'*50, file=f)
    print(abc.contents_pres.value_counts(normalize=True).sort_index(), file=f)
    print('-'*50, file=f)
print('-'*50)
print(abc.toxicity.value_counts(normalize=True).sort_index())
print('-'*50)
print(abc.fluency.value_counts(normalize=True).sort_index())
print('-'*50)
print(abc.contents_pres.value_counts(normalize=True).sort_index())
print('-'*50)