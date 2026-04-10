import pandas as pd

# fpath = '/home/hyeryung/data/mucoco/new_module/em_training/nli/evaluation_results/exp_results_20241127_locate_added.csv'
fpath = '/home/hyeryung/data/mucoco/new_module/em_training/nli/evaluation_results/exp_results_20241127_locate_epr_added.csv'
data = pd.read_csv(fpath, index_col=None)
print(data.head())

def trimmed_mean(df):
    return (df.sum() - df.max() - df.min()) / (len(df) - 2)

data = data.drop(columns=['run_id', 'seed', 'epoch', 'step', 'setting','time_key','run_id+criterion'], errors='ignore')
print('---')
print(data.head())

data_mean = data.groupby(['loss','criterion']).mean().round(4).reset_index()
data_std = data.groupby(['loss','criterion']).std().round(4).reset_index()
data_min = data.groupby(['loss','criterion']).min().round(4).reset_index()
data_max = data.groupby(['loss','criterion']).max().round(4).reset_index()
data_tmean = data.groupby(['loss','criterion']).apply(trimmed_mean).round(4).reset_index()

data_mean['stats'] = 'mean'
data_std['stats'] = 'std'
data_min['stats'] = 'min'
data_max['stats'] = 'max'
data_tmean['stats'] = 'trimmed_mean'

data_all = pd.concat([data_mean, data_std, data_min, data_max, data_tmean], axis=0)

data_all.to_csv('/home/hyeryung/data/mucoco/new_module/em_training/nli/evaluation_results/exp_results_20241127_locate_epr_added_stats.csv', index=False)



