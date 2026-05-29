import os
import datetime
from io import StringIO
import pandas as pd

input_root_dir = '/home/hyeryung/data/loc_edit/models/nli/'
# metrics_file_name = 'epr_snli_locate_metrics.csv'
metrics_file_name = 'nli_contra_300_locate_metrics.csv'
output_save_name = f"{metrics_file_name.split('.csv')[0]}_{datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')}.csv"
output_save_dir = '/home/hyeryung/data/mucoco/laser_edit/ebm_training/nli/evaluation_results'

data_all = []
file_count = 0
for (root, dirs, files) in os.walk(input_root_dir):
    if metrics_file_name in files:
        print(os.path.join(root, metrics_file_name))
        data = pd.read_csv(os.path.join(root, metrics_file_name),index_col=None)
        data_all.append(data)
        file_count+=1

data_all = pd.concat(data_all,ignore_index=True)
data_all.iloc[:, 3:] = data_all.iloc[:, 3:].round(4)
print('--file_count--:',file_count)
data_all.to_csv(os.path.join(output_save_dir,output_save_name),index=False)