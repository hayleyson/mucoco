# ## sentiment

# import glob 
# import pandas as pd
# import numpy as np

# pos_wids = ['2xn81iv5', 'ri45lvy5']
# neg_wids = ['i5k22fxq', '7xrpn8x6']


# for wid in pos_wids:
#     print(wid)
#     fpath = glob.glob(f'/data/hyeryung/mucoco/outputs/sentiment/final/{wid}/outputs_*.txt.intermediate')
#     if len(fpath) == 1:
#         fpath = fpath[0]
#     else:
#         print(fpath)
#         break
    
#     intermed_outputs = pd.read_json(fpath, lines=True)
#     intermed_outputs = intermed_outputs.explode('generations',ignore_index=True)
#     tmpckeys = intermed_outputs.generations.apply(lambda x: list(x.keys()))
#     ckeys = list(set(sum(tmpckeys.tolist(),[])))
    
#     for ckey in ckeys:
#         intermed_outputs[ckey] = intermed_outputs.generations.apply(lambda x: x.get(ckey, np.nan))
    
#     select_cols = [col for col in intermed_outputs.columns if 'best_text' in col]
#     # print(select_cols)
#     intermed_outputs_ = intermed_outputs[select_cols]
#     # print(intermed_outputs_)
    
#     print("percentiles")
#     print(intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).quantile(np.arange(0, 1.1, 0.1)))
#     print(f"mean num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).mean()}")
#     print(f"median num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).median()}")
#     print(f"min num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).min()}")
#     print(f"max num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).max()}")
#     # print(len(intermed_outputs_.isna().sum(axis=1)))
#     print("-"*50)
    
    
# for wid in neg_wids:
#     print(wid)
#     fpath = glob.glob(f'/data/hyeryung/mucoco/outputs/sentiment/final/{wid}/outputs_*.txt.intermediate')
#     if len(fpath) == 1:
#         fpath = fpath[0]
#     else:
#         print(fpath)
#         break
    
#     intermed_outputs = pd.read_json(fpath, lines=True)
#     intermed_outputs = intermed_outputs.explode('generations',ignore_index=True)
#     tmpckeys = intermed_outputs.generations.apply(lambda x: list(x.keys()))
#     ckeys = list(set(sum(tmpckeys.tolist(),[])))
    
#     for ckey in ckeys:
#         intermed_outputs[ckey] = intermed_outputs.generations.apply(lambda x: x.get(ckey, np.nan))
    
#     select_cols = [col for col in intermed_outputs.columns if 'best_text' in col]
#     # print(select_cols)
#     intermed_outputs_ = intermed_outputs[select_cols]
#     # print(intermed_outputs_)
    
#     print("percentiles")
#     print(intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).quantile(np.arange(0, 1.1, 0.1)))
#     print(f"mean num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).mean()}")
#     print(f"median num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).median()}")
#     print(f"min num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).min()}")
#     print(f"max num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).max()}")
#     # print(len(intermed_outputs_.isna().sum(axis=1)))
#     print("-"*50)


## toxicity

# import glob 
# import pandas as pd
# import numpy as np

# pos_wids = ['4kp4ti6s', '6p3qdx0z', 'sutt25w1']


# for wid in pos_wids:
#     print(wid)
#     fpath = glob.glob(f'/data/hyeryung/mucoco/outputs/toxicity/final/{wid}/outputs_*.txt.intermediate')
#     if len(fpath) == 1:
#         fpath = fpath[0]
#     else:
#         print(fpath)
#         break
    
#     intermed_outputs = pd.read_json(fpath, lines=True)
#     intermed_outputs = intermed_outputs.explode('generations',ignore_index=True)
#     tmpckeys = intermed_outputs.generations.apply(lambda x: list(x.keys()))
#     ckeys = list(set(sum(tmpckeys.tolist(),[])))
    
#     for ckey in ckeys:
#         intermed_outputs[ckey] = intermed_outputs.generations.apply(lambda x: x.get(ckey, np.nan))
    
#     select_cols = [col for col in intermed_outputs.columns if 'best_text' in col]
#     # print(select_cols)
#     intermed_outputs_ = intermed_outputs[select_cols]
#     # print(intermed_outputs_)
    
#     print("percentiles")
#     print(intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).quantile(np.arange(0, 1.1, 0.1)))
#     print(f"mean num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).mean()}")
#     print(f"median num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).median()}")
#     print(f"min num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).min()}")
#     print(f"max num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).max()}")
#     # print(len(intermed_outputs_.isna().sum(axis=1)))
#     print("-"*50)
    
## formality

import glob 
import pandas as pd
import numpy as np

pos_wids = ['39uampje', 'cutgmg96', 'wyq1n2po']
neg_wids = ['17oyxgsn', 'pe45pmd4', 'l65c0nw2']


for wid in pos_wids:
    print(wid)
    fpath = glob.glob(f'/data/hyeryung/mucoco/outputs/formality/final/{wid}/outputs_*.txt.intermediate')
    if len(fpath) == 1:
        fpath = fpath[0]
    else:
        print(fpath)
        break
    
    intermed_outputs = pd.read_json(fpath, lines=True)
    intermed_outputs = intermed_outputs.explode('generations',ignore_index=True)
    tmpckeys = intermed_outputs.generations.apply(lambda x: list(x.keys()))
    ckeys = list(set(sum(tmpckeys.tolist(),[])))
    
    for ckey in ckeys:
        intermed_outputs[ckey] = intermed_outputs.generations.apply(lambda x: x.get(ckey, np.nan))
    
    select_cols = [col for col in intermed_outputs.columns if 'best_text' in col]
    # print(select_cols)
    intermed_outputs_ = intermed_outputs[select_cols]
    # print(intermed_outputs_)
    
    print("percentiles")
    print(intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).quantile(np.arange(0, 1.1, 0.1)))
    print(f"mean num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).mean()}")
    print(f"median num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).median()}")
    print(f"min num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).min()}")
    print(f"max num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).max()}")
    # print(len(intermed_outputs_.isna().sum(axis=1)))
    print("-"*50)
    
    
for wid in neg_wids:
    print(wid)
    fpath = glob.glob(f'/data/hyeryung/mucoco/outputs/formality/final/{wid}/outputs_*.txt.intermediate')
    if len(fpath) == 1:
        fpath = fpath[0]
    else:
        print(fpath)
        continue
    
    intermed_outputs = pd.read_json(fpath, lines=True)
    intermed_outputs = intermed_outputs.explode('generations',ignore_index=True)
    tmpckeys = intermed_outputs.generations.apply(lambda x: list(x.keys()))
    ckeys = list(set(sum(tmpckeys.tolist(),[])))
    
    for ckey in ckeys:
        intermed_outputs[ckey] = intermed_outputs.generations.apply(lambda x: x.get(ckey, np.nan))
    
    select_cols = [col for col in intermed_outputs.columns if 'best_text' in col]
    # print(select_cols)
    intermed_outputs_ = intermed_outputs[select_cols]
    # print(intermed_outputs_)
    
    print("percentiles")
    print(intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).quantile(np.arange(0, 1.1, 0.1)))
    print(f"mean num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).mean()}")
    print(f"median num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).median()}")
    print(f"min num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).min()}")
    print(f"max num iterations to converge: {intermed_outputs_.isna().sum(axis=1).apply(lambda x: 10-x).max()}")
    # print(len(intermed_outputs_.isna().sum(axis=1)))
    print("-"*50)