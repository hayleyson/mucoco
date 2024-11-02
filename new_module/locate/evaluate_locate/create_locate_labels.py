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

def get_char_labels(row, contradict_phrase_col: str, text_col: str):
    contradict_phrase = list(filter(lambda x: [x_item for x_item in x if x_item != ''], row[contradict_phrase_col].split('•')))
    text = row[text_col]
    char_label = np.array([0 for _ in range(len(text))])

    contradict_phrase_match = []
    for phrase in contradict_phrase:
        curr_contradict_phrase_match = list(re.finditer(phrase, text))
        if len(curr_contradict_phrase_match) > 1:
            raise ValueError(f"Multiple matches found for {phrase} in text")
        elif len(curr_contradict_phrase_match) == 0:
            raise ValueError(f"No match found for {phrase} in text")
        for match in curr_contradict_phrase_match:
            char_label[match.start():match.end()] = 1
            
    return char_label

def get_char_labels_mnli(row, contradict_phrase_col: str, text_col: str):
    contradict_phrase = list(filter(lambda x: [x_item for x_item in x if x_item != ''], row[contradict_phrase_col].split('•')))
    text = row[text_col]
    char_label = np.array([0 for _ in range(len(text))])

    contradict_phrase_match = []
    for phrase in contradict_phrase:
        curr_contradict_phrase_match = list(re.finditer(phrase, text))
        if len(curr_contradict_phrase_match) > 1:
            print(f"Multiple matches found for {phrase} in text")
            print(f"pairID: {row['pairID']}")
            print(f"text: {row[text_col]}")
            continue
            # raise ValueError(f"Multiple matches found for {phrase} in text")
        elif len(curr_contradict_phrase_match) == 0:
            raise ValueError(f"No match found for {phrase} in text")
        
        for match in curr_contradict_phrase_match:
            char_label[match.start():match.end()] = 1
            
    return char_label

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

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

def handle_snli():
    print('Processing SNLI dataset...')

    print('1. Adding pairID and gold_label.')
    data1 = pd.read_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/snli_annotation/annotator1_snli.jsonl', lines=True).reset_index(drop=True)
    data2 = pd.read_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/snli_annotation/annotator2_snli.jsonl', lines=True).reset_index(drop=True)
    data3 = pd.read_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/snli_annotation/annotator3_snli.jsonl', lines=True).reset_index(drop=True)

    assert set(data2['snli_id']) == set(data2['snli_id']) == set(data3['snli_id']), 'snli_id is not matched'

    snli_test = pd.read_json('/data/hyeryung/mucoco/data/nli/snli_1.0/snli_1.0_test.jsonl', lines=True)
    snli_test['split'] = 'test'
    snli_train = pd.read_json('/data/hyeryung/mucoco/data/nli/snli_1.0/snli_1.0_train.jsonl', lines=True)
    snli_train['split'] = 'train'
    snli_dev = pd.read_json('/data/hyeryung/mucoco/data/nli/snli_1.0/snli_1.0_dev.jsonl', lines=True)
    snli_dev['split'] = 'dev'
    snli_all = pd.concat([snli_train, snli_dev, snli_test], ignore_index=True)

    for data in [data1, data2, data3]:
        case_a, case_b, case_c = [], [], []
        for i, row in data.iterrows():
            
            premise_phrases = [x.strip() for x in row['UP'].split('•') + row['EP'].split('•') + row['NP'].split('•') + row['CP'].split('•') if x != '']
            hypothesis_phrases = [x.strip() for x in row['UH'].split('•') + row['EH'].split('•') + row['NH'].split('•') + row['CH'].split('•') if x != '']
            
            premise_match = snli_all['sentence1'].apply(lambda x: check_phrases_existence(premise_phrases, x))
            hypothesis_match = snli_all['sentence2'].apply(lambda x: check_phrases_existence(hypothesis_phrases, x))
            
            match = snli_all.loc[premise_match & hypothesis_match, :].copy()
            
            if match.shape[0] == 0:
                case_a.append(i)
                print(f"i: {i}")
                print(f"premise_phrase: {premise_phrases}")
                print(f"hypothesis_phrase: {hypothesis_phrases}")
                print(f"No matching sentence found in SNLI dataset.")
                print('-'*50)
            elif match.shape[0] == 1:
                data.loc[i,'premise'] = match['sentence1'].values[0]
                data.loc[i,'hypothesis'] = match['sentence2'].values[0]
                data.loc[i, 'pairID'] = match['pairID'].values[0]
                data.loc[i, 'gold_label'] = match['gold_label'].values[0]
                case_b.append(i)
            else:
                premise_match = snli_all['sentence1'].apply(lambda x: check_phrase_exact_match(premise_phrases, x))
                hypothesis_match = snli_all['sentence2'].apply(lambda x: check_phrase_exact_match(hypothesis_phrases, x))
                match2 = snli_all.loc[premise_match & hypothesis_match, :].copy()
                if match2.shape[0] == 1:
                    data.loc[i,'premise'] = match2['sentence1'].values[0]
                    data.loc[i,'hypothesis'] = match2['sentence2'].values[0]
                    data.loc[i, 'pairID'] = match2['pairID'].values[0]
                    data.loc[i, 'gold_label'] = match2['gold_label'].values[0]
                else:
                    case_c.append(i)
                    print(f"i: {i}")
                    print(f"premise_phrase: {premise_phrases}")
                    print(f"hypothesis_phrase: {hypothesis_phrases}")
                    print(f"Multiple matching sentences found in SNLI dataset:")
                    print(match.loc[:, ['sentence1', 'sentence2']])
                    print('-'*50)
                    
    # 해결 안된 케이스는 수동으로 해결 
    data2.loc[54, 'premise'] = snli_all.loc[563972, 'sentence1']
    data2.loc[54, 'hypothesis'] = snli_all.loc[563972, 'sentence2']
    data2.loc[54, 'pairID'] = snli_all.loc[563972, 'pairID']
    data2.loc[54, 'gold_label'] = snli_all.loc[563972, 'gold_label']

    data2.loc[14, 'premise'] = snli_all.loc[564079, 'sentence1']
    data2.loc[14, 'hypothesis'] = snli_all.loc[564079, 'sentence2']
    data2.loc[14, 'pairID'] = snli_all.loc[564079, 'pairID']
    data2.loc[14, 'gold_label'] = snli_all.loc[564079, 'gold_label']

    data3.loc[54, 'premise'] = snli_all.loc[563972, 'sentence1']
    data3.loc[54, 'hypothesis'] = snli_all.loc[563972, 'sentence2']
    data3.loc[54, 'pairID'] = snli_all.loc[563972, 'pairID']
    data3.loc[54, 'gold_label'] = snli_all.loc[563972, 'gold_label']

    data3.loc[14, 'premise'] = snli_all.loc[564079, 'sentence1']
    data3.loc[14, 'hypothesis'] = snli_all.loc[564079, 'sentence2']
    data3.loc[14, 'pairID'] = snli_all.loc[564079, 'pairID']
    data3.loc[14, 'gold_label'] = snli_all.loc[564079, 'gold_label']
    
    assert set(data1['pairID']) == set(data2['pairID']) == set(data3['pairID']), 'pairID is not matched'
    assert data1.loc[data1['gold_label'] == 'contradiction', :].shape[0] == data2.loc[data2['gold_label'] == 'contradiction', :].shape[0] == data3.loc[data3['gold_label'] == 'contradiction', :].shape[0]

    data1.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/snli_annotation/annotator1_snli_pairID.jsonl', orient='records', lines=True)
    data2.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/snli_annotation/annotator2_snli_pairID.jsonl', orient='records', lines=True)
    data3.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/snli_annotation/annotator3_snli_pairID.jsonl', orient='records', lines=True)

    # ------------------------------------------------------------------------------------------------
    print('2. Adding char, token, word-level labels')
    # get char-level labels
    data1['premise_char_labels'] = data1.apply(lambda x: get_char_labels(x, 'CP', 'premise'), axis=1)
    data1['hypothesis_char_labels'] = data1.apply(lambda x: get_char_labels(x, 'CH', 'hypothesis'), axis=1)
    data2['premise_char_labels'] = data2.apply(lambda x: get_char_labels(x, 'CP', 'premise'), axis=1)
    data2['hypothesis_char_labels'] = data2.apply(lambda x: get_char_labels(x, 'CH', 'hypothesis'), axis=1)
    data3['premise_char_labels'] = data3.apply(lambda x: get_char_labels(x, 'CP', 'premise'), axis=1)
    data3['hypothesis_char_labels'] = data3.apply(lambda x: get_char_labels(x, 'CH', 'hypothesis'), axis=1)

    # merge char-level labels of three datasets
    col_subset = ['pairID', 'premise_char_labels', 'hypothesis_char_labels']
    merged_labels = data1[col_subset].merge(data2[col_subset], on=['pairID'], how='inner', suffixes=('_1', '_2')).merge(data3[col_subset].rename(columns={'premise_char_labels': 'premise_char_labels_3', 'hypothesis_char_labels': 'hypothesis_char_labels_3'}), on=['pairID'], how='inner')
    assert len(merged_labels) == len(data1) == len(data2) == len(data3)

    merged_labels['premise_char_labels'] = merged_labels.apply(lambda x: np.round((x['premise_char_labels_1'] + x['premise_char_labels_2'] + x['premise_char_labels_3']) / 3., 2), axis=1)
    merged_labels['hypothesis_char_labels'] = merged_labels.apply(lambda x: np.round((x['hypothesis_char_labels_1'] + x['hypothesis_char_labels_2'] + x['hypothesis_char_labels_3']) / 3., 2), axis=1)
    merged_labels = merged_labels[['pairID', 'premise_char_labels', 'hypothesis_char_labels']]

    # add premise and hypothesis
    merged_labels = pd.merge(merged_labels, data1[['pairID', 'premise', 'hypothesis', 'gold_label']], on='pairID', how='inner')

    # add word & token level labels
    merged_labels['hypothesis_words'] = merged_labels['hypothesis'].str.split()
    merged_labels['premise_words'] = merged_labels['premise'].str.split()
    merged_labels['hypothesis_tokens'] = merged_labels['hypothesis'].apply(lambda x: tokenizer(x, add_special_tokens=False)['input_ids'])
    merged_labels['premise_tokens'] = merged_labels['premise'].apply(lambda x: tokenizer(x, add_special_tokens=False)['input_ids'])

    merged_labels['hypothesis_word2char'] = merged_labels.apply(lambda x: get_word2char(x, ' ', 'hypothesis_words'), axis=1)
    merged_labels['hypothesis_char2word']=merged_labels['hypothesis_word2char'].apply(kv_swap)
    merged_labels['hypothesis_word_labels']=merged_labels.apply(lambda x: char_label_to_word_label(x, char_labels_col="hypothesis_char_labels", char2word_col="hypothesis_char2word", word_col="hypothesis_words"),axis=1)

    merged_labels['hypothesis_tok2char'] = merged_labels.apply(lambda x: get_tok2char(x, 'hypothesis_tokens', 'hypothesis'), axis=1)
    merged_labels['hypothesis_char2tok']=merged_labels['hypothesis_tok2char'].apply(kv_swap)
    merged_labels['hypothesis_tokens_labels']=merged_labels.apply(lambda x: char_label_to_token_label(x, char_labels_col="hypothesis_char_labels", char2tok_col="hypothesis_char2tok", tokens_col="hypothesis_tokens"),axis=1)

    merged_labels['premise_word2char'] = merged_labels.apply(lambda x: get_word2char(x, ' ', 'premise_words'), axis=1)
    merged_labels['premise_char2word']=merged_labels['premise_word2char'].apply(kv_swap)
    merged_labels['premise_word_labels']=merged_labels.apply(lambda x: char_label_to_word_label(x, char_labels_col="premise_char_labels", char2word_col="premise_char2word", word_col="premise_words"),axis=1)

    merged_labels['premise_tok2char'] = merged_labels.apply(lambda x: get_tok2char(x, 'premise_tokens', 'premise'), axis=1)
    merged_labels['premise_char2tok']=merged_labels['premise_tok2char'].apply(kv_swap)
    merged_labels['premise_tokens_labels']=merged_labels.apply(lambda x: char_label_to_token_label(x, char_labels_col="premise_char_labels", char2tok_col="premise_char2tok", tokens_col="premise_tokens"),axis=1)

    merged_labels['hypothesis_word2tok'] = merged_labels.apply(lambda x: get_word2tok(x, 'hypothesis_tokens', 'hypothesis_words'), axis=1)
    merged_labels['hypothesis_tok2word'] = merged_labels['hypothesis_word2tok'].apply(kv_swap)
    
    merged_labels = merged_labels[['pairID', 'premise', 'hypothesis', 'gold_label', 'hypothesis_words', 'premise_words', 'hypothesis_tokens', 'premise_tokens', 'hypothesis_word2tok', 'hypothesis_tok2word',
        'premise_char_labels', 'hypothesis_char_labels', 'hypothesis_word_labels', 'hypothesis_tokens_labels', 'premise_word_labels', 'premise_tokens_labels']]

    merged_labels.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/snli_annotation/snli_locate_labels.jsonl', orient='records', lines=True)

def handle_mnli():
    print('Processing MNLI dataset...')
    print('1. Adding pairID and gold_label.')
    data1 = pd.read_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/mnli_annotation/annotator1_mnli_matched.jsonl', lines=True).reset_index(drop=True)
    data2 = pd.read_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/mnli_annotation/annotator2_mnli_matched.jsonl', lines=True).reset_index(drop=True)
    data3 = pd.read_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/mnli_annotation/annotator3_mnli_matched.jsonl', lines=True).reset_index(drop=True)

    assert set(data2['snli_id']) == set(data2['snli_id']) == set(data3['snli_id']), 'snli_id is not matched'

    mnli_dev_matched = pd.read_json('/data/hyeryung/mucoco/data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl', lines=True)
    snli_all = mnli_dev_matched

    for data in [data1, data2, data3]:
        case_a, case_b, case_c = [], [], []
        for i, row in data.iterrows():
            
            premise_phrases = [x.strip() for x in (row['UP'].split('•') + row['EP'].split('•') + row['NP'].split('•') + row['CP'].split('•')) if x != '']
            hypothesis_phrases = [x.strip() for x in (row['UH'].split('•') + row['EH'].split('•') + row['NH'].split('•') + row['CH'].split('•')) if x != '']
            
            premise_match = snli_all['sentence1'].apply(lambda x: check_phrases_existence(premise_phrases, x))
            hypothesis_match = snli_all['sentence2'].apply(lambda x: check_phrases_existence(hypothesis_phrases, x))
            
            match = snli_all.loc[premise_match & hypothesis_match, :].copy()
            
            if match.shape[0] == 0:
                case_a.append(i)
                print(f"i: {i}")
                print(f"premise_phrase: {premise_phrases}")
                print(f"hypothesis_phrase: {hypothesis_phrases}")
                print(f"No matching sentence found in SNLI dataset.")
                print('-'*50)
            elif match.shape[0] == 1:
                data.loc[i,'premise'] = match['sentence1'].values[0]
                data.loc[i,'hypothesis'] = match['sentence2'].values[0]
                data.loc[i, 'pairID'] = match['pairID'].values[0]
                data.loc[i, 'gold_label'] = match['gold_label'].values[0]
                case_b.append(i)
            else:
                premise_match = snli_all['sentence1'].apply(lambda x: check_phrase_exact_match(premise_phrases, x))
                hypothesis_match = snli_all['sentence2'].apply(lambda x: check_phrase_exact_match(hypothesis_phrases, x))
                match2 = snli_all.loc[premise_match & hypothesis_match, :].copy()
                if match2.shape[0] == 1:
                    data.loc[i,'premise'] = match2['sentence1'].values[0]
                    data.loc[i,'hypothesis'] = match2['sentence2'].values[0]
                    data.loc[i, 'pairID'] = match2['pairID'].values[0]
                    data.loc[i, 'gold_label'] = match2['gold_label'].values[0]
                else:
                    case_c.append(i)
                    print(f"i: {i}")
                    print(f"premise_phrase: {premise_phrases}")
                    print(f"hypothesis_phrase: {hypothesis_phrases}")
                    print(f"Multiple matching sentences found in SNLI dataset:")
                    print(match.loc[:, ['sentence1', 'sentence2']])
                    print('-'*50)

    # 해결 안된 케이스는 수동으로 해결 
    data1.loc[36, 'premise'] = snli_all.loc[8250, 'sentence1']
    data1.loc[36, 'hypothesis'] = snli_all.loc[8250, 'sentence2']
    data1.loc[36, 'pairID'] = snli_all.loc[8250, 'pairID']
    data1.loc[36, 'gold_label'] = snli_all.loc[8250, 'gold_label']

    # note: data3 에서 7, 16 번째 케이스는 라벨링이 안되었음 -> 임의로 부여
    # set(data1['pairID']) - set(data3['pairID'])
    data3.loc[7, 'pairID'] = '106013c'
    data3.loc[7, 'premise'] = snli_all.loc[snli_all['pairID'] == '106013c', 'sentence1'].values[0]
    data3.loc[7, 'hypothesis'] = snli_all.loc[snli_all['pairID'] == '106013c', 'sentence2'].values[0]
    data3.loc[7, 'gold_label'] = snli_all.loc[snli_all['pairID'] == '106013c', 'gold_label'].values[0]

    data3.loc[16, 'pairID'] = '35878c'
    data3.loc[16, 'premise'] = snli_all.loc[snli_all['pairID'] == '35878c', 'sentence1'].values[0]
    data3.loc[16, 'hypothesis'] = snli_all.loc[snli_all['pairID'] == '35878c', 'sentence2'].values[0]
    data3.loc[16, 'gold_label'] = snli_all.loc[snli_all['pairID'] == '35878c', 'gold_label'].values[0]

    assert set(data1['pairID']) == set(data2['pairID']) == set(data3['pairID']), 'pairID is not matched'
    assert data1.loc[data1['gold_label'] == 'contradiction', :].shape[0] == data2.loc[data2['gold_label'] == 'contradiction', :].shape[0] == data3.loc[data3['gold_label'] == 'contradiction', :].shape[0]

    data1.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/mnli_annotation/annotator1_mnli_matched_pairID.jsonl', orient='records', lines=True)
    data2.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/mnli_annotation/annotator2_mnli_matched_pairID.jsonl', orient='records', lines=True)
    data3.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/mnli_annotation/annotator3_mnli_matched_pairID.jsonl', orient='records', lines=True)

    # ------------------------------------------------------------------------------------------------
    print('2. Adding char, token, word-level labels')

    # get char-level labels
    data1['premise_char_labels'] = data1.apply(lambda x: get_char_labels_mnli(x, 'CP', 'premise'), axis=1)
    data1['hypothesis_char_labels'] = data1.apply(lambda x: get_char_labels_mnli(x, 'CH', 'hypothesis'), axis=1)
    data2['premise_char_labels'] = data2.apply(lambda x: get_char_labels_mnli(x, 'CP', 'premise'), axis=1)
    data2['hypothesis_char_labels'] = data2.apply(lambda x: get_char_labels_mnli(x, 'CH', 'hypothesis'), axis=1)
    data3['premise_char_labels'] = data3.apply(lambda x: get_char_labels_mnli(x, 'CP', 'premise'), axis=1)
    data3['hypothesis_char_labels'] = data3.apply(lambda x: get_char_labels_mnli(x, 'CH', 'hypothesis'), axis=1)

    # multiple match 인 경우에는 수동으로 처리 
    curr_pairID = '4676c'; curr_phrase = 'yeah'
    curr_label = data1.loc[data1['pairID'] == curr_pairID, 'premise_char_labels'].values[0]
    match = list(re.finditer(curr_phrase, data1.loc[data1['pairID'] == curr_pairID, 'premise'].values[0]))[0]
    curr_label[match.start():match.end()] = 1
    data1.at[data.loc[(data1['pairID'] == curr_pairID),:].index[0], 'premise_char_labels'] = curr_label

    curr_pairID = '36766c'; curr_phrase = 'smiled'
    curr_label = data1.loc[data1['pairID'] == curr_pairID, 'premise_char_labels'].values[0]
    match = list(re.finditer(curr_phrase, data1.loc[data1['pairID'] == curr_pairID, 'premise'].values[0]))[0]
    curr_label[match.start():match.end()] = 1
    data1.at[data.loc[(data1['pairID'] == curr_pairID),:].index[0], 'premise_char_labels'] = curr_label

    curr_pairID = '4676c'; curr_phrase = 'yeah'
    curr_label = data2.loc[data2['pairID'] == curr_pairID, 'premise_char_labels'].values[0]
    match = list(re.finditer(curr_phrase, data2.loc[data2['pairID'] == curr_pairID, 'premise'].values[0]))[0]
    curr_label[match.start():match.end()] = 1
    data2.at[data.loc[(data2['pairID'] == curr_pairID),:].index[0], 'premise_char_labels'] = curr_label

    curr_pairID = '36766c'; curr_phrase = 'smiled'
    curr_label = data2.loc[data2['pairID'] == curr_pairID, 'premise_char_labels'].values[0]
    match = list(re.finditer(curr_phrase, data2.loc[data2['pairID'] == curr_pairID, 'premise'].values[0]))[0]
    curr_label[match.start():match.end()] = 1
    data2.at[data.loc[(data2['pairID'] == curr_pairID),:].index[0], 'premise_char_labels'] = curr_label

    curr_pairID = '4676c'; curr_phrase = 'yeah'
    curr_label = data3.loc[data3['pairID'] == curr_pairID, 'premise_char_labels'].values[0]
    match = list(re.finditer(curr_phrase, data3.loc[data3['pairID'] == curr_pairID, 'premise'].values[0]))[0]
    curr_label[match.start():match.end()] = 1
    data3.at[data.loc[(data3['pairID'] == curr_pairID),:].index[0], 'premise_char_labels'] = curr_label

    # ## 아까 data3 에서 7, 16 번째 케이스는 라벨링이 안되었음 -> char_labels 가 어떻게 들어가있는지 체크 
    # print(data3.loc[7, ['pairID', 'premise', 'hypothesis', 'gold_label', 'premise_char_labels', 'hypothesis_char_labels']])
    # print(data3.loc[7, 'premise_char_labels'].sum())
    # print(data3.loc[7, 'hypothesis_char_labels'].sum())
    # print(data3.loc[16, ['pairID', 'premise', 'hypothesis', 'gold_label', 'premise_char_labels', 'hypothesis_char_labels']])
    # print(data3.loc[16, 'premise_char_labels'].sum())
    # print(data3.loc[16, 'hypothesis_char_labels'].sum())

    # merge char-level labels of three datasets
    col_subset = ['pairID', 'premise_char_labels', 'hypothesis_char_labels']
    merged_labels = data1[col_subset].merge(data2[col_subset], on=['pairID'], how='inner', suffixes=('_1', '_2')).merge(data3[col_subset].rename(columns={'premise_char_labels': 'premise_char_labels_3', 'hypothesis_char_labels': 'hypothesis_char_labels_3'}), on=['pairID'], how='inner')
    assert len(merged_labels) == len(data1) == len(data2) == len(data3)

    merged_labels['premise_char_labels'] = merged_labels.apply(lambda x: np.round((x['premise_char_labels_1'] + x['premise_char_labels_2'] + x['premise_char_labels_3']) / 3., 2), axis=1)
    merged_labels['hypothesis_char_labels'] = merged_labels.apply(lambda x: np.round((x['hypothesis_char_labels_1'] + x['hypothesis_char_labels_2'] + x['hypothesis_char_labels_3']) / 3., 2), axis=1)
    merged_labels = merged_labels[['pairID', 'premise_char_labels', 'hypothesis_char_labels']]

    ## annotator3의 라벨이 없었던 부분 수동으로 처리 
    curr_labels = merged_labels.at[merged_labels.loc[merged_labels['pairID'] == '106013c', :].index[0], 'hypothesis_char_labels']
    curr_labels[curr_labels == 0.33] = 0.5
    curr_labels[curr_labels == 0.67] = 1.0
    merged_labels.at[merged_labels.loc[merged_labels['pairID'] == '106013c', :].index[0], 'hypothesis_char_labels'] = curr_labels

    curr_labels = merged_labels.at[merged_labels.loc[merged_labels['pairID'] == '35878c', :].index[0], 'hypothesis_char_labels']
    curr_labels[curr_labels == 0.33] = 0.5
    curr_labels[curr_labels == 0.67] = 1.0
    merged_labels.at[merged_labels.loc[merged_labels['pairID'] == '35878c', :].index[0], 'hypothesis_char_labels'] = curr_labels

    # add premise and hypothesis
    merged_labels = pd.merge(merged_labels, data1[['pairID', 'premise', 'hypothesis', 'gold_label']], on='pairID', how='inner')

    # add word & token level labels
    merged_labels['hypothesis_words'] = merged_labels['hypothesis'].str.split()
    merged_labels['premise_words'] = merged_labels['premise'].str.split()
    merged_labels['hypothesis_tokens'] = merged_labels['hypothesis'].apply(lambda x: tokenizer(x, add_special_tokens=False)['input_ids'])
    merged_labels['premise_tokens'] = merged_labels['premise'].apply(lambda x: tokenizer(x, add_special_tokens=False)['input_ids'])

    merged_labels['hypothesis_word2char'] = merged_labels.apply(lambda x: get_word2char(x, ' ', 'hypothesis_words'), axis=1)
    merged_labels['hypothesis_char2word']=merged_labels['hypothesis_word2char'].apply(kv_swap)
    merged_labels['hypothesis_word_labels']=merged_labels.apply(lambda x: char_label_to_word_label(x, char_labels_col="hypothesis_char_labels", char2word_col="hypothesis_char2word", word_col="hypothesis_words"),axis=1)

    merged_labels['hypothesis_tok2char'] = merged_labels.apply(lambda x: get_tok2char(x, 'hypothesis_tokens', 'hypothesis'), axis=1)
    merged_labels['hypothesis_char2tok']=merged_labels['hypothesis_tok2char'].apply(kv_swap)
    merged_labels['hypothesis_tokens_labels']=merged_labels.apply(lambda x: char_label_to_token_label(x, char_labels_col="hypothesis_char_labels", char2tok_col="hypothesis_char2tok", tokens_col="hypothesis_tokens"),axis=1)

    merged_labels['premise_word2char'] = merged_labels.apply(lambda x: get_word2char(x, ' ', 'premise_words'), axis=1)
    merged_labels['premise_char2word']=merged_labels['premise_word2char'].apply(kv_swap)
    ## 에러가 나는 부분 수동으로 처리 : 55번째 샘플 premise가 "oh  that's accommodating" 라 띄어쓰기가 2개 연속
    merged_labels.at[55, 'premise_char2word'] = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2,
                                                12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2}
    merged_labels['premise_word_labels']=merged_labels.apply(lambda x: char_label_to_word_label(x, char_labels_col="premise_char_labels", char2word_col="premise_char2word", word_col="premise_words"),axis=1)

    merged_labels['premise_tok2char'] = merged_labels.apply(lambda x: get_tok2char(x, 'premise_tokens', 'premise'), axis=1)
    merged_labels['premise_char2tok']=merged_labels['premise_tok2char'].apply(kv_swap)
    merged_labels['premise_tokens_labels']=merged_labels.apply(lambda x: char_label_to_token_label(x, char_labels_col="premise_char_labels", char2tok_col="premise_char2tok", tokens_col="premise_tokens"),axis=1)

    merged_labels['hypothesis_word2tok'] = merged_labels.apply(lambda x: get_word2tok(x, 'hypothesis_tokens', 'hypothesis_words'), axis=1)
    merged_labels['hypothesis_tok2word'] = merged_labels['hypothesis_word2tok'].apply(kv_swap)

    merged_labels = merged_labels[['pairID', 'premise', 'hypothesis', 'gold_label', 'hypothesis_words', 'premise_words', 'hypothesis_tokens', 'premise_tokens', 'hypothesis_word2tok', 'hypothesis_tok2word',
        'premise_char_labels', 'hypothesis_char_labels', 'hypothesis_word_labels', 'hypothesis_tokens_labels', 'premise_word_labels', 'premise_tokens_labels']]

    merged_labels.to_json('/data/hyeryung/mucoco/new_module/data/EPR/text_file/mnli_annotation/mnli_matched_locate_labels.jsonl', orient='records', lines=True)

if __name__ == '__main__':
    handle_snli()
    handle_mnli()