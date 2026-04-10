"""
2024/11/27
Code to merge labels collected from 3 annotators for NLI-locate task
The dataset used for labelling is 100 contradiction samples from snli test, 100 contradiction samples from mnli dev (matched & mismatched), and 100 from anli test.
For mnli & anli, I conducted stratified sampling based on the genre distribution.
"""

import os
os.chdir('/home/hyeryung/data/mucoco')


import random
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
import re

def check_phrases_existence(phrases, text, mode='all'):
    if mode == 'all':
        for phrase in phrases:
            if phrase not in text:
                return False
        return True
    else:
        raise NotImplementedError

def check_phrase_exact_match(phrases, text):
    for phrase in phrases:
        if phrase in text:
            text = text.replace(phrase, '')
        else:
            return False
    if (text.strip() == '') or (text.strip() == '.'): # if after removing all phrases, the text is empty, then return True
        return True
    else:
        return False


def get_tok2char(row: pd.Series, tokens_col: str, text_col: str) -> dict:
    """
    A function to convert a list of tokens into a mapping between each token's index and its corresponding character offsets.
    @param row: A row from dataframe
    @return tok2char: A dictionary with token's location index as keys and tuples of corresponding character offsets as values.

    Example:
    row=pd.Series()
    row['text']='wearing games and holy ****ing shit do I hate horse wearing games .'
    row['tokens']=[86, 6648, 1830, 290, 11386, 25998, 278, 7510, 466, 314, 5465, 8223, 5762, 1830, 764]
    tok2char=get_tok2char(row)
    tok2char
    {0: (0,),
    1: (1, 2, 3, 4, 5, 6),
    2: (7, 8, 9, 10, 11, 12),
    3: (13, 14, 15, 16),
    ...
    13: (59, 60, 61, 62, 63, 64),
    14: (65,66)}
    """
    global tokenizer
    tok2char=dict()
    token_offsets=[0]
    j = 0
    for i in range(1,len(row[tokens_col])+1):
        while True:
            if tokenizer.decode(tokenizer.encode(row[text_col][:j],add_special_tokens=False)) != tokenizer.decode(row[tokens_col][:i]):
                if tokenizer.decode(row[tokens_col][:i])[-1]=='�':#handle a case where a character is split into multiple tokens
                    break
                j+=1
            else:
                token_offsets.append(j)
                tok2char[i-1]=tuple(range(token_offsets[-2],token_offsets[-1]))
                tmp_id = i-2
                while (tmp_id >= 0 and tmp_id not in tok2char):
                    tok2char[tmp_id]=tuple(range(token_offsets[-2],token_offsets[-1]))
                    tmp_id-=1
                j+=1
                break
    return tok2char

def get_word2char(row: pd.Series, ws: str, words_col: str) -> dict:
    """
    A function to convert a list of words into a mapping between each word's index and its corresponding character offsets.
    @param row: A row from dataframe
    @return word2char: A dictionary with word's location index as keys and tuples of corresponding character offsets as values.

    Caveat:
    This code assumes that words are separated by only one type of whitespace, e.g. space.

    Example:
    row=pd.Series()
    row['words']=['wearing', 'games', 'and', 'holy', '****ing', 'shit', 'do', 'I', 'hate', 'horse', 'wearing', 'games.']
    word2char=get_word2char(row)
    word2char
    {0: (0, 1, 2, 3, 4, 5, 6),
    1: (7, 8, 9, 10, 11, 12),...
    9: (45, 46, 47, 48, 49, 50),
    10: (51, 52, 53, 54, 55, 56, 57, 58),
    11: (59, 60, 61, 62, 63, 64, 65)}
    """
    
    word_offsets=[0]
    word2char=dict()
    for i in range(1,len(row[words_col])+1):
        decoded=ws.join(row[words_col][:i])
        word_offsets.append(len(decoded))
        word2char[i-1]=tuple(range(word_offsets[i-1],word_offsets[i]))
        
    return word2char

def get_word2tok(row: pd.Series, tokens_col: str, words_col: str, ws: str=None) -> dict:
    """
    A function that take a list of words and a corresponding list of tokens 
    into a mapping between each word's index and its corresponding token indexes.
    @param row: A row from dataframe
    @return word2char: A dictionary with word's location index as keys and tuples of corresponding token location indexes as values.

    Example:
    row=pd.Series()
    row['words']=['wearing', 'games', 'and', 'holy', '****ing', 'shit', 'do', 'I', 'hate', 'horse', 'wearing', 'games.']
    row['tokens']=[86, 6648, 1830, 290, 11386, 25998, 278, 7510, 466, 314, 5465, 8223, 5762, 1830, 13]
    word2tok=get_word2tok(row)
    word2tok
    {0: [0, 1],
    1: [2],
    2: [3],
    ...
    10: [12],
    11: [13, 14]}
    """
    global tokenizer
    
    jl, jr, k = 0, 0, 0
    grouped_tokens = []
    if ws is not None:
        while jr <= len(row[tokens_col])+1 and k < len(row[words_col]):
            # print(f"{jl}, {jr}, {k}: {tokenizer.decode(row[tokens_col][jl:jr]).strip(' ')}")
            if tokenizer.decode(row[tokens_col][jl:jr]).strip(' ') == row[words_col][k]:
                grouped_tokens.append(list(range(jl,jr)))
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1
        word2tok = dict(zip(range(len(grouped_tokens)), grouped_tokens))
    else:
        while jr <= len(row[tokens_col])+1 and k < len(row[words_col]):
            # print(f"{jl}, {jr}, {k}: {tokenizer.decode(row[tokens_col][jl:jr]).strip()}")
            if tokenizer.decode(row[tokens_col][jl:jr]).strip() == row[words_col][k]:
                grouped_tokens.append(list(range(jl,jr)))
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1
        word2tok = dict(zip(range(len(grouped_tokens)), grouped_tokens))
    return word2tok

def kv_swap(x):

    return_dict=dict()
    for k,v in x.items():
        for item in v:
            return_dict[item]=k
    return return_dict

def get_char_labels(row, label_col: str, text_col: str) -> np.array:
    sent = row[text_col]
    label = row[label_col]
    sent_char_label = np.array([0 for _ in range(len(sent))])
    if type(label) != list: ## if label is missing
        return sent_char_label
    if len(label) > 0:
        for phrase in label:
            sent_char_label[phrase['start']:phrase['end']] = [1 for _ in range(phrase['end']-phrase['start'])]
    return sent_char_label


def char_label_to_word_label(row, char_labels_col:str, char2word_col:str, word_col:str):
    word_labels = [0 for _ in range(len(row[word_col]))]
    for i, lab in enumerate(row[char_labels_col]):
        if lab > 0:
            word_labels[row[char2word_col][i]] = lab
    return word_labels

def char_label_to_token_label(row, char_labels_col:str, char2tok_col:str, tokens_col:str):
    token_labels = [0 for _ in range(len(row[tokens_col]))]
    for i, lab in enumerate(row[char_labels_col]):
        if lab > 0:
            token_labels[row[char2tok_col][i]] = lab
    return token_labels

def word_label_to_char_label(row, text_col, word2char_col, word_label_col) -> np.array:
    char_label = np.array([0 for _ in range(len(row[text_col]))])
    for id, val in enumerate(row[word_label_col]):
        if val == 1:
            for char_id in row[word2char_col][id]:
                char_label[char_id] = 1
    return char_label

tokenizer = AutoTokenizer.from_pretrained("roberta-large")


def handle_nli_contra_300_data():
    # Read data & preprocess

    ## treat data gathered using label-studio 
    ## create char_level labels from label, label2

    data1 = pd.read_json('new_module/data/NLI_locate/raw_results/project-1-at-2024-11-24-04-21-5ffa789b_user1.json').reset_index(drop=True)
    data2 = pd.read_json('new_module/data/NLI_locate/raw_results/project-1-at-2024-11-25-13-09-685517d1.json').reset_index(drop=True)
    data3 = pd.read_json('new_module/data/NLI_locate/raw_results/project-7-at-2024-11-27-09-55-a7f8f21e.json').reset_index(drop=True)

    # check for missing labels : it's okay if the missing label's in premise
    for i, chck_data in enumerate([data1,data2,data3]):
        print(f'----data {i+1}----')
        print('# na in premise label:', chck_data['label'].isna().sum())
        print('# na in hypothesis label:', chck_data['label2'].isna().sum())

    data1['premise_char_labels'] = data1.apply(lambda x: get_char_labels(x, 'label', 'sentence1'), axis=1)
    data1['hypothesis_char_labels'] = data1.apply(lambda x: get_char_labels(x, 'label2', 'sentence2'), axis=1)
    data2['premise_char_labels'] = data2.apply(lambda x: get_char_labels(x, 'label', 'sentence1'), axis=1)
    data2['hypothesis_char_labels'] = data2.apply(lambda x: get_char_labels(x, 'label2', 'sentence2'), axis=1)
    data3['premise_char_labels'] = data3.apply(lambda x: get_char_labels(x, 'label', 'sentence1'), axis=1)
    data3['hypothesis_char_labels'] = data3.apply(lambda x: get_char_labels(x, 'label2', 'sentence2'), axis=1)

    ## for SNLI data for annotator1, I gathered the data via excel sheet. treat it separately and then concat.
    ## special treatment for data1_2
    data1_2 = pd.read_excel('new_module/data/NLI_locate/raw_results/snli_test_contra_labelling_sheet_user1.xlsx',index_col=0,sheet_name='Sheet1')
    data1_2_sents = data1_2.loc[data1_2['level_1'].isin(['sentence1','sentence2']), ['pairID','level_1', 0]].copy()
    data1_2_labels = data1_2.loc[data1_2['level_1'].isin(['sentence1_labels', 'sentence2_labels'])].copy()

    data1_2_sents = pd.pivot(data1_2_sents,index='pairID',columns='level_1',values=0).reset_index()
    data1_2_sents['sentence1_words'] =data1_2_sents['sentence1'].str.split() 
    data1_2_sents['sentence2_words'] =data1_2_sents['sentence2'].str.split() 

    data1_2_labels['list_var'] = data1_2_labels.loc[:, 0:].apply(lambda x: [val for val in x], axis=1)
    data1_2_labels = pd.pivot(data1_2_labels,index='pairID',columns='level_1',values='list_var').reset_index()

    data1_2 = pd.merge(data1_2_sents, data1_2_labels,on='pairID',how='inner')

    def process_labels(row, label_col, words_col):
        labels = row[label_col]
        labels = labels[:len(row[words_col])]
        labels = [0 if pd.isna(x) else x for x in labels]
        return labels

    data1_2['sentence1_labels'] = data1_2.apply(lambda x: process_labels(x, 'sentence1_labels', 'sentence1_words'), axis=1)
    data1_2['sentence2_labels'] = data1_2.apply(lambda x: process_labels(x, 'sentence2_labels', 'sentence2_words'), axis=1)

    ## change the label from word-level to char level 
    data1_2['sentence1_word2char'] = data1_2.apply(lambda x: get_word2char(x, ' ', 'sentence1_words'), axis=1)
    data1_2['sentence2_word2char'] = data1_2.apply(lambda x: get_word2char(x, ' ', 'sentence2_words'), axis=1)

    data1_2['premise_char_labels'] = data1_2.apply(lambda x: word_label_to_char_label(x, 'sentence1','sentence1_word2char', 'sentence1_labels'), axis=1)
    data1_2['hypothesis_char_labels'] = data1_2.apply(lambda x: word_label_to_char_label(x, 'sentence2','sentence2_word2char', 'sentence2_labels'), axis=1)

    ## select only necessary columns
    data1_2['source'] = 'snli' # add source column
    data1_2 = data1_2[['pairID', 'source','sentence1', 'sentence2', 'premise_char_labels', 'hypothesis_char_labels']].reset_index(drop=True)

    # check for missing labels : it's okay if the missing label's in premise
    for i, chck_data in enumerate([data1_2]):
        print(f'----data {i+1} SNLI ----')
        print('# na in premise label:', (chck_data['premise_char_labels'].apply(sum) == 0).sum())
        print('# na in hypothesis label:', (chck_data['hypothesis_char_labels'].apply(sum) == 0).sum())

    ## concat data1 and data1_2 
    data1 = pd.concat([data1, data1_2],axis=0,ignore_index=True)

    # -------------------------- #
    ## c.f. duplicates in pairID? -> no
    assert len(set(data1['pairID'])) == len(data1)
    assert len(set(data2['pairID'])) == len(data2)
    assert len(set(data3['pairID'])) == len(data3)
    # -------------------------- #

    # Merge three labeler's char-level annotations

    ## merge char-level labels of three datasets
    col_subset = ['pairID', 'source', 'premise_char_labels', 'hypothesis_char_labels']
    merge_cols = ['pairID', 'source']
    merged_labels = data1[col_subset].merge(data2[col_subset], on=merge_cols, how='inner', suffixes=('_1', '_2')).merge(data3[col_subset].rename(columns={'premise_char_labels': 'premise_char_labels_3', 'hypothesis_char_labels': 'hypothesis_char_labels_3'}), on=merge_cols, how='inner')
    assert len(merged_labels) == len(data1) == len(data2) == len(data3)

    merged_labels['premise_char_labels'] = merged_labels.apply(lambda x: np.round((x['premise_char_labels_1'] + x['premise_char_labels_2'] + x['premise_char_labels_3']) / 3., 2), axis=1)
    merged_labels['hypothesis_char_labels'] = merged_labels.apply(lambda x: np.round((x['hypothesis_char_labels_1'] + x['hypothesis_char_labels_2'] + x['hypothesis_char_labels_3']) / 3., 2), axis=1)
    merged_labels = merged_labels[['pairID', 'premise_char_labels', 'hypothesis_char_labels']]

    ## add premise and hypothesis
    merged_labels = pd.merge(merged_labels, data1[['pairID', 'source','sentence1', 'sentence2']], on='pairID', how='inner')
    merged_labels = merged_labels.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'})
    merged_labels.loc[:,'gold_label'] = 'contradiction' # add gold_label just for the sake of formatting

    # Convert char_level annotations to word/token-level annotations

    ## add word & token level labels
    merged_labels['hypothesis_words'] = merged_labels['hypothesis'].str.split()
    merged_labels['premise_words'] = merged_labels['premise'].str.split()
    merged_labels['hypothesis_tokens'] = merged_labels['hypothesis'].apply(lambda x: tokenizer(x, add_special_tokens=False)['input_ids'])
    merged_labels['premise_tokens'] = merged_labels['premise'].apply(lambda x: tokenizer(x, add_special_tokens=False)['input_ids'])

    merged_labels['hypothesis_word2char'] = merged_labels.apply(lambda x: get_word2char(x, ' ', 'hypothesis_words'), axis=1)

    ## error when space comes at the beggining of sentence or there are double spaces in the middle of sentence.
    ## treat manually.

    # space at the beggining -> 0th word should contain 7 characters. other words indices should be shifted by 1. last word should contain up to 58.
    hypothesis_word2char_row_45 = {0: (0, 1, 2, 3, 4, 5, 6),
    1: (7, 8, 9, 10),
    2: (11, 12, 13, 14),
    3: (15, 16, 17, 18, 19),
    4: (20, 21, 22, 23, 24, 25, 26),
    5: (27, 28, 29, 30, 31, 32, 33, 34, 35),
    6: (36, 37, 38, 39, 40, 41),
    7: (42, 43, 44, 45, 46),
    8: (47, 48, 49, 50, 51),
    9: (52, 53, 54, 55, 56, 57, 58)}

    hypothesis_word2char_row_44 = {0: (0, 1, 2, 3, 4, 5, 6, 7, 8),
    1: (9, 10, 11, 12, 13),
    2: (14, 15, 16, 17, 18, 19),
    3: (20, 21, 22),
    4: (23, 24, 25),
    5: (26, 27, 28, 29, 30, 31, 32, 33),
    6: (34, 35, 36, 37, 38, 39),
    7: (40, 41, 42, 43, 44),
    8: (45, 46, 47, 48),
    9: (49, 50, 51, 52, 53, 54, 55, 56, 57),
    10: (58, 59, 60),
    11: (61, 62, 63, 64, 65, 66),
    12: (67, 68, 69),
    13: (70, 71, 72, 73, 74, 75, 76, 77),
    14: (78, 79, 80, 81),
    15: (82, 83),
    16: (84, 85, 86),
    17: (87, 88, 89, 90),
    18: (91, 92, 93, 94, 95, 96), ## two spaces in front of U.S.
    19: (97, 98, 99, 100, 101, 102, 103, 104, 105, 106),
    20: (107, 108, 109, 110),
    21: (111, 112, 113, 114, 115)} ## one space after 200

    hypothesis_word2char_row_26 = {0: (0, 1, 2),
    1: (3, 4, 5, 6, 7, 8, 9),
    2: (10, 11, 12, 13, 14, 15, 16, 17, 18),
    3: (19, 20, 21),
    4: (22, 23, 24, 25, 26),
    5: (27, 28, 29, 30, 31),
    6: (32, 33, 34, 35, 36, 37, 38), ## two space in front of known
    7: (39, 40, 41),
    8: (42, 43, 44, 45),
    9: (46, 47, 48, 49, 50),
    10: (51, 52, 53, 54, 55, 56, 57, 58),
    11: (59, 60, 61, 62, 63, 64, 65),
    12: (66, 67, 68, 69, 70, 71, 72),
    13: (73, 74, 75, 76, 77)}

    merged_labels.loc[[26], 'hypothesis_word2char'] = [hypothesis_word2char_row_26]
    merged_labels.loc[[44], 'hypothesis_word2char'] = [hypothesis_word2char_row_44]
    merged_labels.loc[[45], 'hypothesis_word2char'] = [hypothesis_word2char_row_45]

    merged_labels['hypothesis_char2word']=merged_labels['hypothesis_word2char'].apply(kv_swap)
    merged_labels['hypothesis_word_labels']=merged_labels.apply(lambda x: char_label_to_word_label(x, char_labels_col="hypothesis_char_labels", char2word_col="hypothesis_char2word", word_col="hypothesis_words"),axis=1)

    merged_labels['hypothesis_tok2char'] = merged_labels.apply(lambda x: get_tok2char(x, 'hypothesis_tokens', 'hypothesis'), axis=1)
    merged_labels['hypothesis_char2tok']=merged_labels['hypothesis_tok2char'].apply(kv_swap)
    merged_labels['hypothesis_token_labels']=merged_labels.apply(lambda x: char_label_to_token_label(x, char_labels_col="hypothesis_char_labels", char2tok_col="hypothesis_char2tok", tokens_col="hypothesis_tokens"),axis=1)

    merged_labels['premise_word2char'] = merged_labels.apply(lambda x: get_word2char(x, ' ', 'premise_words'), axis=1)

    ## error when space comes at the beggining of sentence or there are double spaces in the middle of sentence.
    ## treat manually.

    # ## double space in the middle, one space at the end
    premise_word2char_row_190 = {0: (0, 1, 2, 3),
    1: (4, 5, 6, 7, 8, 9, 10),
    2: (11, 12, 13, 14, 15),
    3: (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
    4: (31, 32, 33),
    5: (34, 35, 36, 37),
    6: (38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    7: (52, 53, 54, 55, 56, 57, 58, 59, 60),
    8: (61, 62, 63, 64, 65), ## two spaces in front of and
    9: (66, 67, 68, 69, 70, 71, 72, 73, 74),
    10: (75, 76, 77, 78, 79, 80, 81, 82),
    11: (83, 84),
    12: (85, 86, 87, 88, 89),
    13: (90, 91, 92, 93, 94, 95)}

    merged_labels.loc[[190], 'premise_word2char'] = [premise_word2char_row_190]
    merged_labels['premise_char2word']=merged_labels['premise_word2char'].apply(kv_swap)
    merged_labels['premise_word_labels']=merged_labels.apply(lambda x: char_label_to_word_label(x, char_labels_col="premise_char_labels", char2word_col="premise_char2word", word_col="premise_words"),axis=1)


    merged_labels['premise_tok2char'] = merged_labels.apply(lambda x: get_tok2char(x, 'premise_tokens', 'premise'), axis=1)
    merged_labels['premise_char2tok']=merged_labels['premise_tok2char'].apply(kv_swap)
    merged_labels['premise_token_labels']=merged_labels.apply(lambda x: char_label_to_token_label(x, char_labels_col="premise_char_labels", char2tok_col="premise_char2tok", tokens_col="premise_tokens"),axis=1)


    # add mapping from words to tokens and vice versa
    merged_labels['hypothesis_word2tok'] = merged_labels.apply(lambda x: get_word2tok(x, 'hypothesis_tokens', 'hypothesis_words'), axis=1)
    merged_labels['hypothesis_tok2word'] = merged_labels['hypothesis_word2tok'].apply(kv_swap)


    # add binarized labels for hypothesis
    merged_labels["hypothesis_word_labels_binary"] = merged_labels["hypothesis_word_labels"].apply(lambda x: [1 if y>=0.5 else 0 for y in x])
    merged_labels["hypothesis_token_labels_binary"] = merged_labels["hypothesis_token_labels"].apply(lambda x: [1 if y>=0.5 else 0 for y in x])


    merged_labels = merged_labels[['pairID', 'premise', 'hypothesis', 'gold_label', 'source', 'hypothesis_words', 'premise_words', 'hypothesis_tokens', 'premise_tokens', 'hypothesis_word2tok', 'hypothesis_tok2word',
        'premise_char_labels', 'premise_word_labels', 'premise_token_labels', 
        'hypothesis_char_labels', 'hypothesis_word_labels', 'hypothesis_word_labels_binary', 'hypothesis_token_labels', 'hypothesis_token_labels_binary']]

    merged_labels.to_json('new_module/data/NLI_locate/nli_contra_300_locate_labels.jsonl', orient='records', lines=True)


if __name__ == '__main__':
    
    handle_nli_contra_300_data()