import pandas as pd
import sqlalchemy
import sqlite3
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    con = sqlite3.connect("./data/ACL-2014-irony/ironate.db")
    cur = con.cursor()

    # irony_comment = pd.read_sql_query("SELECT * FROM irony_comment",con,dtype=str)
    irony_commentsegment = pd.read_sql_query("SELECT * FROM irony_commentsegment",con,dtype=str)
    irony_label = pd.read_sql_query("SELECT * FROM irony_label", con, dtype=str)

    irony_label = irony_label.loc[irony_label['segment_id']!='nan'].copy()
    irony_label['segment_id'] = irony_label['segment_id'].astype(float).astype(int).astype(str)
    irony_label['label'] = irony_label['label'].apply(lambda x: 0 if x== '-1' else 1)

    irony_label_agg = irony_label.groupby(['comment_id', 'segment_id'])['label'].agg(['mean', 'count']).reset_index()
    irony_label_agg_text = pd.merge(irony_label_agg, irony_commentsegment, left_on=['comment_id', 'segment_id'], right_on=['comment_id', 'id'],how='left',suffixes=['label','comment_segment'])

    # create aggregate label 
    irony_label_agg_text = irony_label_agg_text.drop(columns=['id', 'segment_index'])
    irony_label_agg_text = irony_label_agg_text.rename(columns = {'mean': 'labels'})
    irony_label_agg_text.to_json("./data/ACL-2014-irony/irony_dataset.jsonl", orient="records", lines=True)

    # train:valid split
    train, val_test = train_test_split(irony_label_agg_text, stratify=irony_label_agg_text['labels'], test_size=0.2, random_state=1234)

    # NOTE. label 0.2 has one example => train_test_split(..., stratify=..['labels']) will throw an error 
    # => thus, we will isolate the example, split the rest of the dataset, and add the example to the test set.
    temp = val_test.loc[val_test['labels']==0.2].copy()
    val_test = val_test.loc[val_test['labels']!=0.2].copy()
    val, test = train_test_split(val_test, stratify=val_test['labels'], test_size=0.5, random_state=1234)
    test = pd.concat([test, temp], axis=0)

    print(f"train set size: {train.shape[0]}")
    print(f"validation set size: {val.shape[0]}")
    print(f"test set size: {test.shape[0]}")

    train.to_json("./data/ACL-2014-irony/train.jsonl", orient="records", lines=True)
    val.to_json("./data/ACL-2014-irony/val.jsonl", orient="records", lines=True)
    test.to_json("./data/ACL-2014-irony/test.jsonl", orient="records", lines=True)
    
    # binarize (ver1 - dropping 0.5)
    train_binarized = train.loc[train['labels']!=0.5, ].copy()
    print(f"train set size without samples with label value of 0.5: {train_binarized.shape[0]}")
    train_binarized['labels'] = train_binarized['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    
    val_binarized = val.loc[val['labels']!=0.5, ].copy()
    print(f"val set size without samples with label value of 0.5: {val_binarized.shape[0]}")
    val_binarized['labels'] = val_binarized['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    
    test_binarized = test.loc[test['labels']!=0.5, ].copy()
    print(f"test set size without samples with label value of 0.5: {test_binarized.shape[0]}")
    test_binarized['labels'] = test_binarized['labels'].apply(lambda x: 1 if x > 0.5 else 0)
    
    train_binarized.to_json("./data/ACL-2014-irony/train_binary.jsonl", orient="records", lines=True)
    val_binarized.to_json("./data/ACL-2014-irony/val_binary.jsonl", orient="records", lines=True)
    test_binarized.to_json("./data/ACL-2014-irony/test_binary.jsonl", orient="records", lines=True)
    
    # binarize (ver2 - map 0.5 to 1)
    train_binarized['labels'] = train['labels'].apply(lambda x: 1 if x >= 0.5 else 0)
    val_binarized['labels'] = val['labels'].apply(lambda x: 1 if x >= 0.5 else 0)
    test_binarized['labels'] = test['labels'].apply(lambda x: 1 if x >= 0.5 else 0)
    
    train_binarized.to_json("./data/ACL-2014-irony/train_binary_use0.5_as1.jsonl", orient="records", lines=True)
    val_binarized.to_json("./data/ACL-2014-irony/val_binary_use0.5_as1.jsonl", orient="records", lines=True)
    test_binarized.to_json("./data/ACL-2014-irony/test_binary_use0.5_as1.jsonl", orient="records", lines=True)
    