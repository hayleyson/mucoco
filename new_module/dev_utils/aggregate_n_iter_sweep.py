import argparse
from collections import defaultdict
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_path_root', type=str, required=True, help='example: /data/hyeryung/mucoco/outputs/nli/jvw5mv0t/outputs_epsilon0.99.txt')
# parser.add_argument('--path_pattern', type=str, required=True, help='example: /data/hyeryung/mucoco/outputs/nli/jvw5mv0t/outputs_epsilon0.99.txt.%s-results.txt')
args = parser.parse_args()

# path_pattern = '/data/hyeryung/mucoco/outputs/nli/jvw5mv0t/outputs_epsilon0.99.txt.%s-results.txt'
path_pattern = args.output_path_root + '.%s-results.txt'
result_dict = defaultdict(list)
for i in range(10):
    path = path_pattern % str(i)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dat = line.split(',')
            for item in dat:
                key, value = item.split(': ')
                result_dict[key].append(float(value.strip()))
            # print(line)
        # print()

pd.DataFrame(result_dict).reset_index().rename(columns={'index': 'n_iter'}).to_csv(path_pattern.replace('%s-', ''),index=False)