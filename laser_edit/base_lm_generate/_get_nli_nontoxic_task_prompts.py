import pandas as pd
import json 
from pathlib import Path
from datasets import load_dataset



NUM_EXAMPLES = 250
RANDOM_SEED = 44
ROOT_DIR = Path("laser_edit/base_lm_generate")
SAVE_DIR = Path("/home/hyeryung/data/mucoco/laser_edit/data/nli-toxicity")

# 약간 애매하지만 3가지 정도를 해볼까?
# - (V) toxicity가 살짝 높은 sentence들을 rtp에서 추출해서 그것과 관련한 문장을 쓰도록 하는 것 => 애매한 것 같아서 보류
# - (->) jigsaw에서 toxicity가 높고 길이가 긴 코멘트들을 가져와서 그것과 관련한 문장을 쓰도록 하는 것 
# - (ING) socc 데이터셋에서 editorial의 excerpt를 가져오고, 그것과 관련한 문장을 쓰도록 하는 것 


##############################################################################################################
# - (V) toxicity가 살짝 높은 sentence들을 rtp에서 추출해서 그것과 관련한 문장을 쓰도록 하는 것 => 애매한 것 같아서 보류
##############################################################################################################

# with open(ROOT_DIR / "few_shot_prompts.json", "r") as f:
#     few_shot_prompts = json.load(f)
# few_shots_prompts = few_shot_prompts['nli+nontoxic']
# few_shots_prompts_text = [few_shots_prompts[str(i+1)]['text'] for i in range(len(few_shots_prompts))]
# # few_shots_prompts_text = [text.split('Text: ')[1].split('Hypothesis: ')[0].strip()[:20] for text in few_shots_prompts_text]

# # 1. Read all 10k nontoxic prompts and extract samples that are not used in few-shot prompting 
# rtp = pd.read_json("/home/hyeryung/data/mucoco/laser_edit/data/toxicity-avoidance/control-prompts/nontoxic_prompts-10k.jsonl", lines=True)

# rtp['prompt_text'] = rtp['prompt'].apply(lambda x: x['text'])
# rtp['prompt_toxicity'] = rtp['prompt'].apply(lambda x: x['toxicity'])
# rtp['continuation_text'] = rtp['continuation'].apply(lambda x: x['text'])
# rtp['continuation_toxicity'] = rtp['continuation'].apply(lambda x: x['toxicity'])
# rtp['sentence_text'] = (rtp['prompt_text'] + rtp['continuation_text']).str.strip()


# rtp = rtp.loc[~rtp['sentence_text'].apply(lambda x: any(x.startswith(text) for text in few_shots_prompts_text))].copy()
# print(f"Number of samples after removing few-shot prompts: {len(rtp)}")

# ## NLI + Toxicity -> Prompts with high sentence toxicity => Use ChatGPT
# print(f"Number of samples with continuation toxicity between 0.5 and 0.8: {len(rtp.loc[rtp['continuation_toxicity'].between(0.5, 0.8, inclusive='right')].copy())}")
# rtp_toxic = rtp.loc[rtp['continuation_toxicity'].between(0.5, 0.8, inclusive="right")]#.sample(n=NUM_EXAMPLES, random_state=RANDOM_SEED)
# rtp_toxic = rtp_toxic.loc[rtp_toxic['sentence_text'].str.len() >= 150].copy()
# print(f"Number of samples after removing sentences shorter than 150 characters: {len(rtp_toxic)}")
# rtp_toxic = rtp_toxic.sample(n=NUM_EXAMPLES, random_state=RANDOM_SEED)
# rtp_toxic = rtp_toxic[['prompt', 'sentence_text']].copy()
# rtp_toxic['prompt'] = rtp_toxic['sentence_text'].apply(lambda x: {'text': x})
# del rtp_toxic['sentence_text']

# rtp_toxic.to_json(SAVE_DIR / "nontoxic-prompts-toxic-continuations.jsonl", orient="records", lines=True)

##############################################################################################################
# - (ING) socc 데이터셋에서 editorial의 excerpt를 가져오고, 그것과 관련한 문장을 쓰도록 하는 것 
##############################################################################################################

# # to check the length distribution of anli premises 
# anli = pd.read_json("/home/hyeryung/data/mucoco/data/nli/snli_mnli_anli_train_without_finegrained.jsonl", lines=True)
# anli = anli.loc[anli['source'].isin(['anli_R2_train','anli_R1_train', 'anli_R3_train' ])].copy()
# anli['premise_word_count'] = anli['premise'].apply(lambda x: len(x.split()))
# print(anli.premise_word_count.describe())
# # count    139462.000000
# # mean         54.035207
# # std          16.209290
# # min          14.000000
# # 25%          44.000000
# # 50%          53.000000
# # 75%          61.000000
# # max         138.000000
# # Name: premise_word_count, dtype: float64

## start main code.
articles = pd.read_csv("/home/hyeryung/data/mucoco/laser_edit/data/nli-toxicity/socc/raw/gnm_articles.csv")
print(articles.head())
print(articles.columns)
print(articles.shape)

articles['article_text_word_count'] = articles['article_text'].apply(lambda x: len(x.split()))
articles = articles.loc[articles['article_text_word_count'] >= 50].copy() # filter out one article with only 12 words.
articles = articles.sort_values(by="ncomments", ascending=False).head(250).reset_index(drop=True)
articles['article_text'] = articles['article_text'].str.replace('<p>', '').str.replace('</p>', '')

excerpts = []
word_counts = []
for i in range(len(articles)):
    
    row = articles.iloc[i]
    text = row['article_text']
    sentences = text.split('.')
    word_count = 0
    
    excerpt = []
    for sent in sentences:
        word_count += len(sent.split())
        excerpt.append(sent)
        if word_count >= 50:
            break
    excerpts.append('.'.join(excerpt) + '.')
    word_counts.append(word_count)
    
articles['excerpt'] = excerpts
articles['excerpt_word_count'] = word_counts

# print(articles['excerpt_word_count'].describe())

# articles.to_json(SAVE_DIR / "socc_excerpts_top_ncomments_250.jsonl", orient="records", lines=True)
    
comments = pd.read_csv("/home/hyeryung/data/mucoco/laser_edit/data/nli-toxicity/socc/raw/gnm_comments.csv")
print(comments.columns)

# join comments -> select comments that are direct replies to the article (no parent ID)
articles_comments = pd.merge(articles, comments, on='article_id', how='left')
print(articles_comments.shape)
print(articles_comments.parentID.nunique()); print(articles_comments.parentID.value_counts()); print(articles_comments.parentID.isnull().sum())

articles_comments = articles_comments.loc[articles_comments['parentID'].isnull()].copy()
print(articles_comments.article_id.nunique())


articles_comments['comment_word_count'] = articles_comments['comment_text'].apply(lambda x: len(x.split()))
articles_comments = articles_comments.loc[articles_comments['comment_word_count'] >= 10].copy()
print(articles_comments.article_id.nunique())
articles_comments = articles_comments.groupby('article_id').sample(n=10, random_state=RANDOM_SEED).reset_index(drop=True)
# print(articles_comments['comment_word_count'].describe())
# print(articles_comments.loc[articles_comments['comment_word_count'] < 10, ['article_id', 'comment_text']])
# print(articles_comments.loc[articles_comments['comment_word_count'] >= 10, ['article_id', 'comment_text']])
# print(articles_comments.loc[articles_comments['comment_word_count'] >= 10, ['article_id', 'comment_text']])
# articles_comments = articles_comments.loc[articles_comments['negVotes'] > 0].copy()
# print(articles_comments.article_id.nunique())
print(articles_comments.shape)
# print(articles_comments.sort_values(by='comment_word_count', ascending=False).head(10)['comment_text'].values)
# print(articles_comments.groupby('article_id').count().sort_values(by='comment_id', ascending=False))
    
articles_comments['comment_prefix'] = articles_comments['comment_text'].apply(lambda x: ' '.join(x.split(' ')[:10]))
print(articles_comments.head(10)['comment_prefix'].values)
articles_comments.to_json(SAVE_DIR / "socc_gnm_top250_ncomments_10sampled_root_comments.jsonl", orient="records", lines=True)