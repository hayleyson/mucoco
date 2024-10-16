import pandas as pd
import os
import json
from glob import glob
import datetime

time_keys = """1728404380
1728436169
1727850952""".split()

results_all_time_keys = []
for time_key in time_keys:
    result_dir = f'/data/hyeryung/mucoco/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy/{time_key}'
    result_files = glob(f'{result_dir}/*info.txt')
    result_contents = []
    for file in result_files:
        
        with open(file, 'r') as f:
            
            result_content = json.load(f)
            
        criterion = file.split('/')[-1].split('best_model_')[-1].split('_info.txt')[0]
        result_content['criterion'] = criterion
        result_content['time_key'] = time_key
        result_contents.append(result_content)

    results = pd.DataFrame(result_contents)
    results_criterion = results['criterion']
    results_timekey = results['time_key']
    del results['criterion']
    del results['time_key']
    results.insert(0, 'criterion', results_criterion)
    results.insert(0, 'time_key', results_timekey)
    results.to_csv(f'{result_dir}/results.csv', index=False)
    
    results_all_time_keys.append(results)
    
results_all_time_keys = pd.concat(results_all_time_keys)
results_all_time_keys.to_excel(f"new_module/em_training/nli/evaluation_results/nli_energynet_metrics_{datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')}.xlsx", index=False)
    