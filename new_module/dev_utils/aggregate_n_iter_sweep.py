from glob import glob
import argparse
from collections import defaultdict
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_path_root', type=str, required=True, help='example: /home/hyeryung/data/mucoco/outputs/nli/jvw5mv0t/outputs_epsilon0.99.txt')
# parser.add_argument('--path_pattern', type=str, required=True, help='example: /home/hyeryung/data/mucoco/outputs/nli/jvw5mv0t/outputs_epsilon0.99.txt.%s-results.txt')
parser.add_argument('--column_list', type=str, nargs='+', required=True, help='list of columns to save')
args = parser.parse_args()

# path_pattern = '/home/hyeryung/data/mucoco/outputs/nli/jvw5mv0t/outputs_epsilon0.99.txt.%s-results.txt'
path_pattern = args.output_path_root + '.%s-results.txt'
iterations = [int(x.split('.')[-1]) for x in glob(args.output_path_root + ".*") if x.split('.')[-1].isnumeric()]
max_iterations = max(iterations)
result_dict = defaultdict(list)
for i in range(max_iterations+1):
    path = path_pattern % str(i)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dat = line.split(',')
            for item in dat:
                try:
                    key, value = item.split(': ')
                    result_dict[key.strip()].append(float(value.strip()))
                except ValueError as e: # more than one ': 's
                    if 'h1' in item:
                        key_val, h1_val = item.split('h1: ')
                        key, value = key_val.split(': ')
                        result_dict[key.strip()].append(float(value.strip()))
                        result_dict['h1'].append(float(h1_val.strip()))
                    else:
                        raise ValueError(e)
            # print(line)
        # print()

pd.DataFrame(result_dict)[args.column_list].reset_index().rename(columns={'index': 'n_iter'}).to_csv(path_pattern.replace('%s-', ''),index=False)