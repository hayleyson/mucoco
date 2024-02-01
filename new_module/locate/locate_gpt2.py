import os
import sys
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.." )
sys.path.append(project_dir)

import pandas as pd
import numpy as np
from scipy import stats
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from mucoco.utils.locate import locate

def locate_recall(intermediate_df, label_df):
    indice_cols = [col for col in intermediate_df.columns if 'indices' in col]
    intermediate_df['indices'] = [set() for i in range(len(intermediate_df))]
    
    for col in indice_cols:
        intermediate_df['indices'] = intermediate_df['indices'] + intermediate_df[col]
        
    label_df['gt_gt0'] = [set(np.where(np.array(x) > 0)[0]) for x in gt_file]
    label_df['gt_eq1'] = [set(np.where(np.array(x) == 1.0)[0]) for x in gt_file]
    pass

def locate_precision(intermediate_df, label_df):
    indice_cols = [col for col in intermediate_df.columns if 'indices' in col]
    pass

def mrr(out, labels, input = None): #implement mean reciprocal rank
    idx_array = stats.rankdata(-out, axis=1, method='min')
    labels = labels.astype(int)
    rank = np.take_along_axis(idx_array, labels[:, None], axis=1)
    return np.sum(1/rank)

def calculate_rr(_docs_with_score, _labels):
    
    _docs = [x[0].metadata['application_number'] for x in _docs_with_score] # list of application numbers with length same as docs_with_score
    _yn = [1 if x in _labels else 0 for x in _docs] # list of 1,0 with length same as docs_with_score
    _scores = np.array([x[1] for x in _docs_with_score]) # array of shape (length docs_with_score,) of similarity scores
    
    idx_array = stats.rankdata(_scores, method='max')
    idx_array_gold = idx_array[np.where(np.array(_yn)==1)[0]] # only get ranks of indices where application number is in the labels (gold prior arts)
    if len(idx_array_gold) == 0:
        return np.nan
    else:
        return 1 / min(idx_array_gold)
    
    
checkpoint_path = "/shared/s3/lab07/hyeryung/loc_edit/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds-energy-training/step_2800_best_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
config = AutoConfig.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, config=config)
model.to('cuda')

# data_path = "new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl"
# data = pd.read_json(data_path, lines=True)

# prompt_all = []
# gen_all = []
# indices_all = []
# pred_indices_all = []
# pred_indices_all2 = []
# for i, row in data.iterrows():
    
#     prompt = row['prompt']
#     gens = row['generations']
    
#     prompt_all.extend([prompt['text']] * len(gens))
    
#     gen_texts = [gen['text'] for gen in gens]
#     gen_all.extend(gen_texts)
#     gen_locate_labels = [gen.get('locate_labels', []) for gen in gens]
#     indices_all.extend(gen_locate_labels)
    
#     batch = tokenizer(gen_texts, padding=True, truncation=True, return_tensors="pt")
#     batch.to('cuda')
    
#     locate_result = locate(model, tokenizer, batch, max_num_tokens = 6, num_layer=10, unit="word", use_cuda=True)
#     pred_indices_all.extend(locate_result)
    
#     locate_result_reformat = []
#     for j, gen in enumerate(gens):
#         length = batch['attention_mask'][j].sum().item()
#         ## no need to worry about the case where max of locate_result[j] is greater than length
#         # if len(locate_result[j]) > 0 and max(locate_result[j]) >= length:
#         #     print(tokenizer.decode([batch['input_ids'][j][max(locate_result[j])]]))
#         tmp_locate_result = [1 if k in locate_result[j] else 0 for k in range(length)]
#         locate_result_reformat.append(tmp_locate_result)
    
#     pred_indices_all2.extend(locate_result_reformat)
    
# pd.DataFrame({'prompt': prompt_all, 'gen': gen_all, 'indices': indices_all, 'pred_indices': pred_indices_all, 'pred_indices2': pred_indices_all2}).to_json("new_module/data/toxicity-avoidance/testset_gpt2_2500_locate.jsonl", lines=True, orient='records')

# data = pd.read_json("new_module/data/toxicity-avoidance/testset_gpt2_2500_locate.jsonl", lines=True)
# data = data.loc[data['indices'].apply(len) > 0, :].copy()

# data['indices2'] = data['indices'].apply(lambda x: [1 if i >= 0.5 else 0 for i in x])
# data.to_json("new_module/data/toxicity-avoidance/testset_gpt2_2500_locate.jsonl", lines=True, orient='records')

data = pd.read_json("new_module/data/toxicity-avoidance/testset_gpt2_2500_locate.jsonl", lines=True)

def recall_precision(x):
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, j in zip(x['indices2'], x['pred_indices2']):
        if (i == 0) and (j == 0):
            tn += 1
        elif (i == 0) and (j == 1):
            fp += 1
        elif (i == 1) and (j == 0):
            fn += 1
        else:
            tp += 1
            
    return (tp / (tp + fn + 1e-8), tp / (tp + fp + 1e-8))

data['recall_precision'] = data.apply(recall_precision, axis=1)
data['recall'] = data['recall_precision'].apply(lambda x: x[0])
data['precision'] = data['recall_precision'].apply(lambda x: x[1])
data['f1'] = (2 * data['recall'] * data['precision']) / (data['recall'] + data['precision'] + 1e-8)

# data.to_json("new_module/data/toxicity-avoidance/testset_gpt2_2500_locate_metrics.jsonl", lines=True, orient='records')

print(f"avg f1: {data['f1'].mean()}")
print(f"avg recall: {data['recall'].mean()}")
print(f"avg precision: {data['precision'].mean()}")