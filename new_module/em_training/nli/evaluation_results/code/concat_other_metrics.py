import os
os.chdir('/data/hyeryung/loc_edit/models/nli')

import pandas as pd
import json
from glob import glob
import datetime

result_dirs = [
            #    'roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_margin_ranking',
            #    'roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a',
            #    'roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_margin_ranking',
            #    'roberta_large_snli_mnli_anli_train_dev_with_finegrained_original_labels_cross_entropy_n_a',
            #    'roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a'
            'roberta_large_snli_mnli_anli_train_dev_with_finegrained_3class_finegrained_labels_cross_entropy_n_a']
results_all_time_keys = []
for result_dir in result_dirs:
    print('---------')
    print(result_dir)
    dirs = [x for x in os.listdir(result_dir) if os.path.isdir(f'{result_dir}/{x}')]
    print(dirs)
    
    for dir in dirs:
        print('**', dir, '**')
        result_files = glob(f'{result_dir}/{dir}/*info.txt')
        print(result_files)
        result_contents = []
        for file in result_files:
            
            with open(file, 'r') as f:
                
                result_content = json.load(f)
                
            criterion = file.split('/')[-1].split('best_model_')[-1].split('_info.txt')[0]
            result_content['criterion'] = criterion
            result_content['time_key'] = dir.split('/')[-1]
            result_contents.append(result_content)

        results = pd.DataFrame(result_contents)
        results_criterion = results['criterion']
        results_timekey = results['time_key']
        del results['criterion']
        del results['time_key']
        results.insert(0, 'criterion', results_criterion)
        results.insert(0, 'time_key', results_timekey)
        results.insert(0, 'setting', result_dir)
        # results.to_csv(f'{result_dir}/results.csv', index=False)
        
        results_all_time_keys.append(results)
    
results_all_time_keys = pd.concat(results_all_time_keys)
results_all_time_keys = results_all_time_keys.sort_values(['time_key','criterion'])

os.chdir('/data/hyeryung/mucoco/') # come back to mucoco dir
results_all_time_keys.to_excel(f"new_module/em_training/nli/evaluation_results/exp_results_{datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')}.xlsx", index=False)
    

# import pandas as pd
# import os
# import json
# from glob import glob
# import datetime

# time_keys = """1728404380
# 1728436169
# 1727850952""".split()

# results_all_time_keys = []
# for time_key in time_keys:
#     result_dir = f'/data/hyeryung/mucoco/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy/{time_key}'
#     result_files = glob(f'{result_dir}/*info.txt')
#     result_contents = []
#     for file in result_files:
        
#         with open(file, 'r') as f:
            
#             result_content = json.load(f)
            
#         criterion = file.split('/')[-1].split('best_model_')[-1].split('_info.txt')[0]
#         result_content['criterion'] = criterion
#         result_content['time_key'] = time_key
#         result_contents.append(result_content)

#     results = pd.DataFrame(result_contents)
#     results_criterion = results['criterion']
#     results_timekey = results['time_key']
#     del results['criterion']
#     del results['time_key']
#     results.insert(0, 'criterion', results_criterion)
#     results.insert(0, 'time_key', results_timekey)
#     results.to_csv(f'{result_dir}/results.csv', index=False)
    
#     results_all_time_keys.append(results)
    
# results_all_time_keys = pd.concat(results_all_time_keys)
# results_all_time_keys.to_excel(f"new_module/em_training/nli/evaluation_results/nli_energynet_metrics_{datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')}.xlsx", index=False)
    