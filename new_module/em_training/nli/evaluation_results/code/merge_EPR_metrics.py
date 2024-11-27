import os
from io import StringIO
import pandas as pd

data_all = []
for (root, dirs, files) in os.walk('./'):
    if 'EPR_metrics.csv' in files:
        
        data = pd.read_csv(os.path.join(root, 'EPR_metrics.csv'))
        data_all.append(data)

data_all = pd.concat(data_all)
data_all.to_csv('EPR_metrics.csv',index=False)