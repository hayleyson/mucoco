import ast
import gc, json, logging, os, yaml, re
from typing import List, Tuple

from openai import OpenAI
import numpy as np
import pandas as pd
import scipy, torch, evaluate
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
    pipeline,
)


from laser_edit.set_consistency_energy.baselines.baseline_model import baseline_model
from laser_edit.set_consistency_energy.baselines.LLM.lm_loader import lm_loader
from laser_edit.utils.utils import ravel, unravel, unravel_toxicity_data, read_metric_file
from laser_edit.utils.sc_energy_utils import load_sc_energy_model, parse_set_text, set_consistency_dataset


logging.basicConfig(level=os.getenv('LOGGING_LEVEL', 'INFO'), format="%(message)s")
logger = logging.getLogger(__name__)

def conditional_perplexity(generations_df, model, tokenizer, device='cuda', write_file=None, include_trimmed_mean=False):
    perplexities = []
    goodperplexities = []
    # total_nll = 0
    # total_tokens = 0
    
    total_nll = []
    total_tokens = []
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating PPL', mininterval=5):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.prompt['text']
        prompt_is_empty = False
        if prompt in ["", " ", "<|endoftext|>", tokenizer.bos_token]:
            prompt_is_empty = True
        if prompt == "":
            prompt = tokenizer.bos_token if tokenizer.bos_token else " "
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        #if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
        if not prompt_is_empty:
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = [gen['text'] for gen in row['generations']]
        # for gen_ids in generations:
        for gen in generations:

            # full_input_ids = torch.LongTensor([row.prompt['tokens'] + gen_ids]).to(device)
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            # print(f'{prompt}{gen}')
            # print(full_input_ids)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)

            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])

            ppl = np.exp(loss.item())

            # input()
            if ppl < 100:   # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            # if ppl < 1e4:
            perplexities.append(ppl)
            # else:
                # print("ppl values are weirldly large. Check for errors")

            # total_nll += (full_loss - prompt_loss).item()
            # total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            
            total_nll.append((full_loss - prompt_loss).item())
            if (full_input_ids.shape[1] - prompt_input_ids.shape[1]) == 0: ## TODO. need to address this case. corner case: sometimes all tokens are deleted and empty string becomes the final output of editing.
                total_tokens.append(1)
            else:
                total_tokens.append((full_input_ids.shape[1] - prompt_input_ids.shape[1]))
            
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()
            if write_file is not None:
                fout.write(f"{ppl}, {(full_loss - prompt_loss).item()}, {(full_input_ids.shape[1] - prompt_input_ids.shape[1])}\n")
        # input("ok")
    
    # print(np.nanmean(goodperplexities), len(goodperplexities), len(perplexities), g)
    # print(perplexities)
    # return np.nanmean(perplexities), np.exp(total_nll/total_tokens)
    if include_trimmed_mean:
        notna_perplexities = perplexities[~np.isnan(perplexities)]
        return np.nanmean(perplexities), scipy.stats.trim_mean(notna_perplexities, proportiontocut=0.001), np.exp(np.nansum(total_nll)/np.nansum(total_tokens))
        
    else:
        return np.nanmean(perplexities), np.exp(np.nansum(total_nll)/np.nansum(total_tokens))

def perplexity(generations_df, model, tokenizer, device='cuda', write_file=None):
    #TODO spearman correlation between human ppl and model ppl, not needed anymore, check degen ppl calculation, it's different from this.
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating total PPL'):
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        else:
            prompt_loss = 0
        # for every generation conditioned on the prompt
        generations = [gen['text'] for gen in row['generations']]
        for gen in generations:
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            total_nll += (full_loss - prompt_loss).item()
            total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            
            if write_file is not None:
                fout.write(f"{total_nll} {total_tokens}\n")
        
    return np.exp(total_nll/total_tokens)

def fluency_classify(generations_df, output_file=None):

    # score generations and write to sentiment.jsonl
    # classifier = pipeline(model='textattack/roberta-base-CoLA', device=0, use_safetensors=True)
    classifier = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA", use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA", use_safetensors=True)
    classifier.to("cuda")
    
    print("writing outputs to ", str(output_file))
    
    accuracies = []
    all_prediction_labels = []
    all_prediction_scores = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation fluency', mininterval=5):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}' if gen.startswith(' ') else f'{prompt} {gen}')
        
        inputs = tokenizer(sentences_for_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to("cuda")

        try:
            # predictions_for_prompt = classifier(sentences_for_prompt)
            predictions_for_prompt = classifier(**inputs)
            predictions_for_prompt = F.softmax(predictions_for_prompt.logits, dim=-1)
            c_predictions_for_prompt = predictions_for_prompt.argmax(dim=-1)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)

        # prediction_labels = [prediction["label"] for prediction in predictions_for_prompt]
        prediction_labels = c_predictions_for_prompt.tolist()
        all_prediction_labels += prediction_labels
        # prediction_scores = [str(prediction["score"]) if (prediction["label"] == "LABEL_1") else str(1-prediction["score"]) for prediction in predictions_for_prompt]
        prediction_scores = predictions_for_prompt[:, 1].tolist()
        all_prediction_scores += prediction_scores
        
    if output_file is not None:
        with open(output_file, "w") as fout:
            for label, score in zip(all_prediction_labels, all_prediction_scores):
                fout.write(f"{label},{score}\n")

    # accuracy = np.array(all_prediction_labels) == "LABEL_1" ## LABEL_1 is acceptable
    accuracy = np.array(all_prediction_labels) == 1
    accuracy = np.nanmean(accuracy.astype("float32"))
        
    return accuracy


def sentiment_classify(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis', device=0)
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    from pprint import pprint
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
        
        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "POSITIVE")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt:  
                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies)

def sentiment_classify_big(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    print("lalala")
    classifier = pipeline(model='siebert/sentiment-roberta-large-english', device=0)
    # classifier.cuda()
    print("lalala2")
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    accuracies = []
    positive_proba = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
            
        # print(sentences_for_prompt)
        

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "POSITIVE")
            
            if prediction["label"] == "POSITIVE":
                positive_proba.append(prediction["score"])
            else:
                positive_proba.append(1.0-prediction["score"])
                
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        
        if sentiment_file is not None:
            for res in predictions_for_prompt:  
                fo.write(json.dumps(res) + '\n')
        
    # prompt별 accuracy의 평균, prompt별 accuracy의 표준편차, 모든 generation 기준 accuracy, 모든 generation의 positive_proba의 평균
    return np.nanmean(accuracies), np.std(accuracies), np.nanmean(positive_proba)


def sentiment_classify_own2(generations_df, sentiment_file=None, checkpoint_path=None, model_type=None):

    # score generations and write to sentiment.jsonl
    # classifier = pipeline('sentiment-analysis')
    # model_path="/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-sst-2-with-gpt2-large-embeds/checkpoint_best"
    # model_path="/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-textattack-sst-2-with-gpt2-large-embeds-proper/checkpoint_best"
    # model_path="/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-textattack-uncased-sst-2-with-gpt2-large-embeds-proper/checkpoint_best"
    config = AutoConfig.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if model_type == 'RobertaCustomForSequenceClassification':
        classifier_model = RobertaCustomForSequenceClassification.from_pretrained(checkpoint_path, config=config, use_safetensors=True)
    else:
        classifier_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, config=config, use_safetensors=True)
    classifier = TextClassificationPipeline(task="text-classification", model=classifier_model, tokenizer=tokenizer, device=0)
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    accuracies = []
    positive_proba = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
        # print(sentences_for_prompt)
        

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "LABEL_1")
                        
            if prediction["label"] == "LABEL_1":
                positive_proba.append(prediction["score"])
            else:
                positive_proba.append(1.0-prediction["score"])
        
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        
        if sentiment_file is not None:
            for res in predictions_for_prompt: 

                fo.write(json.dumps(res) + '\n')
        
    # prompt별 accuracy의 평균, prompt별 accuracy의 표준편차, 모든 generation의 positive_proba의 평균
    return np.nanmean(accuracies), np.std(accuracies), np.nanmean(positive_proba)

def toxicity_score(generations_df, perspective_file, perspective_rate_limit=5):
    import time

    from googleapiclient import discovery
    from googleapiclient.errors import HttpError

    API_KEY=os.getenv("PERSPECTIVE_API_KEY")
    assert API_KEY != "none", "Please set the API_KEY before proceeding"

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    total_failed = 0
    avg_max_toxicity = []
    avg_toxicity = []
    toxic_probability_p = []
    toxic_probability_s = []

    # if not os.path.exists(perspective_file):
    fout = open(perspective_file, "w")
    # else:
        # raise ValueError("perspective file path already exists")

    last_request_time = -1
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating toxicity via perspective API'):
        prompt = row.prompt['text']
        # generations = [gen['text'] if len(gen['text']) <= 3000 else gen['text'][:3000] for gen in row['generations'] ] 
        allresponses = []
        for genid in range(len(row['generations'])):
            # generations = [gen['text'] for gen in row['generations'][genid:genid+1]]  
            generations = [gen['text'] for gen in row['generations']]  
            # print(sum([len(gen) for gen in generations]))
            generations = [gen if gen != "" else "\n" for gen in generations]


            responses = {f'gen-{i}-{genid}-{j}': None for j in range(len(generations))}
            
            not_done = np.array([1.0/len(generations) for gen in generations])
            def response_callback(request_id, response, exception):
                responses[request_id] = (response, exception)
                if exception is None:
                    not_done[int(request_id.split("-")[-1])] = 0
                if exception is not None:
                    not_done[int(request_id.split("-")[-1])] = 1
                    print(request_id, exception)
            
            while not_done.sum() > 0:
                try:
                    time_since_last_request = time.time() - last_request_time

                    if time_since_last_request < 1:
                        time.sleep(1-time_since_last_request)

                    if not_done.sum() > 1:
                        print(i, "extra_sleep", not_done.sum(), flush=True)
                        time.sleep(1.0)

                    batch_request = client.new_batch_http_request()
                    for j, text in enumerate(generations):
                        analyze_request= {
                            'comment': {'text': text},
                            'requestedAttributes': {"TOXICITY":{}},
                            'spanAnnotations': True,
                            "languages": ["en"],
                        }
                        # print(analyze_request)
                        batch_request.add(client.comments().analyze(body=analyze_request), callback=response_callback, request_id=f"gen-{i}-{genid}-{j}")
                    # print(batch_request)
                    # input()
                    batch_request.execute()
                    last_request_time = time.time()
                    # print(responses)
                    # time.sleep(1.0)
                except Exception as e:
                    print(e)
                    print("sleeping for 60 sec and retrying")
                    time.sleep(60.0)
            allresponses.append(responses)

        json.dump({"allresponses": responses}, fout)
        fout.write("\n")
        max_prob = 0.0
        toxicity_proportion = 0
        for responses in allresponses: 
            # print(responses)
            for req_id, (response, exception) in responses.items():    
                prob = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
                max_prob = max(max_prob, prob)
                toxicity_proportion += int(prob > 0.5)
                
                avg_toxicity.append(prob)
                toxic_probability_s.append(int(prob > 0.5))
                # response[]
                # x = input("break")
        
        avg_max_toxicity.append(max_prob)
        toxic_probability_p.append(int(toxicity_proportion >= 1))

        # if x == "y":
        #     break

    return (np.nanmean(avg_max_toxicity), sum(toxic_probability_p)/len(toxic_probability_p), \
            np.nanmean(avg_toxicity), sum(toxic_probability_s)/len(toxic_probability_s))

def toxicity_score_int(generations_df, toxicity_file, device, checkpoint_path, model_type=None):

    softmax = nn.Softmax(dim=-1)
    config = AutoConfig.from_pretrained(checkpoint_path)    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if model_type == "RobertaCustomForSequenceClassification":
        model = RobertaCustomForSequenceClassification.from_pretrained(checkpoint_path,config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path,config=config)
    model.to(device)
    model.eval()
    
    if toxicity_file is not None:
        fo = open(toxicity_file, 'w')
    
    avg_max_toxicity = []
    avg_toxicity = []
    toxic_probability_p = []
    toxic_probability_s = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation toxicity'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
        
        batch = tokenizer(sentences_for_prompt, padding=True, truncation=True, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                predictions_for_prompt = model(**batch)
                probs = softmax(predictions_for_prompt['logits'])
                torch.cuda.empty_cache()
                predictions_for_prompt = probs[:, 1].tolist()
                gc.collect()
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [float('nan') for i in range(len(sentences_for_prompt))]
        
        max_prob = 0.0
        toxicity_proportion = 0
        for prob in predictions_for_prompt:
            max_prob = max(max_prob, prob)
            toxicity_proportion += int(prob > 0.5)
        
            avg_toxicity.append(prob)
            toxic_probability_s.append(int(prob > 0.5))
        
        avg_max_toxicity.append(max_prob)
        toxic_probability_p.append(int(toxicity_proportion >= 1))
        
        if toxicity_file is not None:
            for res in predictions_for_prompt: 
                fo.write(json.dumps(res) + '\n')
        
    return (np.nanmean(avg_max_toxicity), sum(toxic_probability_p)/len(toxic_probability_p), \
            np.nanmean(avg_toxicity), sum(toxic_probability_s)/len(toxic_probability_s))
    
def nli_score(generations_df, task, write_file, device='cuda'):
    model_paths = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "cross-encoder/nli-roberta-base",
    "cross-encoder/nli-deberta-v3-base"
]
    models = []
    tokenizers = []
    for model_path in model_paths:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        models.append(model)
        tokenizers.append(tokenizer)

    # 클래스별 확률 합산 변수 초기화
    total_entail_prob = 0
    total_neutral_prob = 0
    total_contradiction_prob = 0
    total_count = 0
    total_contradiction_count = 0
    total_entail_count = 0
    total_neutral_count = 0

    results = []
    # 각 row에 대해 NLI 점수 계산
    for _, row in tqdm(generations_df.iterrows(), total=len(generations_df), desc='NLI classifying...', mininterval=5):
        premise = row['prompt']['text'] if task != "comment" else row['prompt']['article_excerpt']
        hypotheses = [gen['text'] for gen in row['generations']]

        # 각 hypothesis에 대해 NLI 평가
        for hypothesis in hypotheses:
            entail_prob_sum = 0
            neutral_prob_sum = 0
            contradiction_prob_sum = 0

            # 각 모델에 대해 예측 수행
            for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
                # remove exact match
                premise = premise.lower()
                hypothesis = hypothesis.lower()
                if premise in hypothesis:
                    hypothesis.replace(premise, "")
                # 토큰화 및 텐서 변환
                inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).squeeze()  # 예측 확률 계산

                max_prob_class = probs.argmax().item()

                if 'ynie' in model_paths[i]:
                    entail_prob_sum += probs[0].item()  # entailment 확률
                    neutral_prob_sum += probs[1].item()    # neutral 확률
                    contradiction_prob_sum += probs[2].item() # contradiction 확률
                else:
                    contradiction_prob_sum += probs[0].item()  # contradiction 확률
                    entail_prob_sum += probs[1].item()     # entailment 확률
                    neutral_prob_sum += probs[2].item() # neutral 확률


            # 각 hypothesis에 대한 모델 평균 확률 계산 및 누적
            entail_prob_avg = entail_prob_sum / len(models)
            neutral_prob_avg = neutral_prob_sum / len(models)
            contradiction_prob_avg = contradiction_prob_sum / len(models)

            total_entail_prob += entail_prob_avg
            total_neutral_prob += neutral_prob_avg
            total_contradiction_prob += contradiction_prob_avg
            if contradiction_prob_avg == max(entail_prob_avg, neutral_prob_avg, contradiction_prob_avg):
                classified_class = "contradiction"
                total_contradiction_count += 1
            elif entail_prob_avg == max(entail_prob_avg, neutral_prob_avg, contradiction_prob_avg):
                classified_class = "entail" 
                total_entail_count += 1
            else:
                classified_class = 'neutral'
                total_neutral_count += 1
            total_count += 1

            results.append({
                "entailment_prob": entail_prob_avg,
                "neutral_prob": neutral_prob_avg,
                "contradiction_prob": contradiction_prob_avg,
                "nli_class": classified_class
            })

    # 전체 데이터에 대한 평균 확률 계산
    avg_nli_entail = total_entail_prob / total_count
    avg_nli_neutral = total_neutral_prob / total_count
    avg_nli_contradiction = total_contradiction_prob / total_count
    entail_ratio = total_entail_count / total_count
    neutral_ratio = total_neutral_count / total_count
    contadiction_ratio = total_contradiction_count / total_count

    if write_file:
        with open(write_file, 'w') as f:
            for result in results:
                f.write(f"{result}\n")

    return avg_nli_entail, avg_nli_neutral, avg_nli_contradiction, contadiction_ratio, entail_ratio, neutral_ratio


def toxicity_nli_joint_rates_from_saved_eval(generations_file_path, toxicity_saved_path=None, nli_saved_path=None, **kw):
    p, r = Path(generations_file_path), Path(generations_file_path).name + "-results.txt"
    d = p.parent
    tox = toxicity_saved_path or str(d / (r + ".toxicity") if (d / (r + ".toxicity")).is_file() else d / (r + ".toxicity_int"))
    nli = nli_saved_path or str(d / (r + ".nli"))
    return toxicity_nli_joint_percentages_from_paths(tox, nli, **kw)


def toxicity_nli_joint_percentages_from_paths(
    toxicity_path: str, nli_path: str, toxic_prob_threshold: float = 0.5
) -> dict:
    """Non-toxic = toxicity prob <= threshold; consistent = NLI class is not contradiction."""
    with open(toxicity_path, encoding="utf-8") as f:
        tox_lines = [ln.strip() for ln in f if ln.strip()]
    if json.loads(tox_lines[0]).get("allresponses"):
        tox_probs = []
        for line in tox_lines:
            d = json.loads(line)["allresponses"]
            pairs = sorted(
                (
                    int(k.rsplit("-", 1)[-1]),
                    float(v[0]["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"]),
                )
                for k, v in d.items()
            )
            tox_probs.extend(p for _, p in pairs)
    else:
        tox_probs = [float(json.loads(x)) for x in tox_lines]

    cons = []
    with open(nli_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cons.append(ast.literal_eval(line.strip())["nli_class"] != "contradiction")

    n = min(len(tox_probs), len(cons))
    if not n:
        return {"n": 0, "pct_both": 0.0, "pct_only_nontoxic": 0.0, "pct_only_consistent": 0.0, "pct_neither": 0.0}
    both = nt = co = ne = 0
    for i in range(n):
        t = tox_probs[i] <= toxic_prob_threshold
        c = cons[i]
        if t and c:
            both += 1
        elif t:
            nt += 1
        elif c:
            co += 1
        else:
            ne += 1
    s = 100.0 / n
    return {
        "n": n,
        "pct_both": both * s,
        "pct_only_nontoxic": nt * s,
        "pct_only_consistent": co * s,
        "pct_neither": ne * s,
    }


def formality_score_ext(generations_df, output_file, device):
    
    def collate_fn(example_batch):
       return tokenizer(example_batch, padding=True, truncation=True, return_tensors="pt").to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/roberta-base-formality")
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/roberta-base-formality")
    model.to(device)
    model.eval()
    
    softmax = nn.Softmax(dim=-1)
    
    generations_df = generations_df.explode('generations')
    generations = generations_df["generations"]
    texts = [example['text'] for example in generations]
    dataset = Dataset.from_list(texts)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=False, collate_fn=collate_fn)
    
    formality_scores = []
    formal_counts = 0
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            # print(outputs.logits)
            probs = softmax(outputs.logits)
            formality_scores.extend(probs[:, -1].tolist())
            formal_counts += torch.sum(torch.where(probs[:,-1] >= 0.5,1,0)).item()
            
    with open(output_file, 'w') as f:
        f.writelines([str(x)+'\n' for x in formality_scores])
    
    return np.nanmean(formality_scores), formal_counts/len(texts)


def formality_score_int(generations_df, output_file, device, checkpoint_path, model_type=None):
    
    def collate_fn(example_batch):
       return tokenizer(example_batch, padding=True, truncation=True, return_tensors="pt").to(device)

    softmax = nn.Softmax(dim=-1)
    config = AutoConfig.from_pretrained(checkpoint_path)    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if model_type == "RobertaCustomForSequenceClassification":
        model = RobertaCustomForSequenceClassification.from_pretrained(checkpoint_path,config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path,config=config)
    model.to(device)
    model.eval()
    
    generations = generations_df["generations"]
    texts = [example[0]['text'] for example in generations]
    dataset = Dataset.from_list(texts)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=False, collate_fn=collate_fn)
    
    formality_scores = []
    formal_counts = 0
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            probs = softmax(outputs.logits)
            formality_scores.extend(probs[:, -1].tolist())
            formal_counts += torch.sum(torch.where(probs[:,-1] >= 0.5,1,0)).item()
            
    with open(output_file, 'w') as f:
        f.writelines([str(x)+'\n' for x in formality_scores])
    
    return np.nanmean(formality_scores), formal_counts/len(texts)

        
        

def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating dist-n'):
        generations = [gen['text'] for gen in row['generations']]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

def distinctness2(generations_df): #not over samples but averaged over individual outputs
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating dist-n'):
        generations = [gen['text'] for gen in row['generations']]
        for gen in generations:
            unigrams, bigrams, trigrams = set(), set(), set()
            total_words = 0
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
            dist1.append(len(unigrams) / total_words)
            dist2.append(len(bigrams) / total_words)
            dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


def repetition(generations_df, tokenizer, numbers_only=True, rep_file=None):
    """
    Proportion of examples with repeated phrases of length 3 or more.
    """
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    objs = []
    max_n = 90

    n_repeated_examples = 0
    total_examples = 0

    if rep_file is not None:
        fout = open(rep_file, "w")
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating repetitions'):
        if 'tokens' not in row['generations'][0]:
            generations = [tokenizer.encode(gen['text'], add_special_tokens=False) for gen in row['generations']]
        else:
            generations = [gen['tokens'] for gen in row['generations']]
        for gen in generations:
            total_examples += 1
            
            if type(gen) == int: ## temporary fix (23.01.13) : for cases where gen is just one token and got squeezed to be an integer.
                gen = [gen]
            if len(gen) == 0:
                continue
            if gen[-1] == SEP:
                gen.pop(-1)
            rev_gen = list(reversed(gen))
            last_n_repeats = [0] * max_n

            for n in range(1, max_n + 1):
                n_repeat = 1
                while len(rev_gen[n*n_repeat:n*(n_repeat+1)]) == n and \
                        rev_gen[n*n_repeat:n*(n_repeat+1)] == rev_gen[:n]:
                    n_repeat += 1
                last_n_repeats[n - 1] = n_repeat
            max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

            if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
                repetition = {
                    'repeated_phrase': list(reversed(rev_gen[:max_repeated_n + 1])),
                    'repeated_times': last_n_repeats[max_repeated_n],
                    'repeated_phrase_length': max_repeated_n + 1,
                }
                n_repeated_examples += 1
            else:
                repetition = {}
            
            if rep_file is not None:
                json.dump(repetition, fout)
                fout.write("\n")
    
    if rep_file is not None:
        fout.close()

    return n_repeated_examples*1.0/total_examples


def _source_df_after_unravel_ravel(sources_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate prompts / rebundle generations via unravel → ravel.

    `unravel` leaves one row per generation with strings in column ``generations``;
    `ravel` expects that string in column ``text`` (see ``laser_edit/utils/utils.py``).
    """
    exploded = unravel(sources_df.copy())
    if "text" not in exploded.columns:
        exploded = exploded.rename(columns={"generations": "text"})
    return ravel(exploded)


def contents_preservation_metrics(sources_file,outputs_df,results_file,task):
    
    if task in ['toxicity','sentiment','set_nli', 'set_snli', 'set_lconvqa', 'lconvqa', 'vqa', 'nli_toxicity']:
        sources = pd.read_json(sources_file, lines=True)
        predictions = outputs_df.copy()

        if task != 'toxicity':
            if len(sources) != len(predictions):
                sources = _source_df_after_unravel_ravel(sources)
            if len(sources) != len(predictions):
                raise ValueError(
                    f"contents_preservation ({task}): source and prediction row counts still differ after "
                    f"source unravel+ravel (sources_file={sources_file!r}, outputs_file={outputs_file!r}): "
                    f"len(sources)={len(sources)} len(predictions)={len(predictions)}."
                )

        sources.prompt=sources.prompt.apply(lambda x: x['text'])
        sources.columns = sources.columns[:1].tolist() + [x+'_source' for x in sources.columns[1:]]
        
        if type(predictions['prompt'].values[0]) == str:
            predictions['prompt'] = predictions['prompt'].apply(lambda x: {'text': x})
        if type(predictions['generations'].values[0][0]) == str:
            predictions['generations'] = predictions['generations'].apply(lambda x: [ {'text': y} for y in x])

        predictions.prompt=predictions.prompt.apply(lambda x: x['text'])
        predictions.columns = predictions.columns[:1].tolist() + [x+'_prediction' for x in predictions.columns[1:]]

        if task == 'toxicity':
            source_predictions = pd.merge(sources, predictions, on='prompt', how='inner')
        else:
            # Row counts match; pair by row index (avoid duplicate ``prompt`` columns from concat).
            source_predictions = pd.concat([sources, predictions.drop(columns=['prompt'])], axis=1)

        prompt_list=[]
        source_list=[]
        prediction_list=[]
        for row_idx, (_, row) in enumerate(source_predictions.iterrows()):
            gs = row.generations_source
            gp = row.generations_prediction
            if not isinstance(gs, list) or not isinstance(gp, list):
                raise ValueError(
                    f"contents_preservation ({task}): row {row_idx} expects list-valued generations "
                    f"(got generations_source={type(gs).__name__}, generations_prediction={type(gp).__name__})."
                )
            if len(gs) != len(gp):
                raise ValueError(
                    f"contents_preservation ({task}): row {row_idx} has len(generations_source)={len(gs)} "
                    f"but len(generations_prediction)={len(gp)}."
                )
            prompt_list.extend([row.prompt]*len(gs))
            for i in range(len(gs)):
                source_list.append(gs[i]['text'])
                prediction_list.append(gp[i]['text'])
        source_predictions_=pd.DataFrame({'prompt':prompt_list,'source':source_list,'prediction':prediction_list})
        
    elif task=='formality':
        with open(sources_file,'r') as f:
            sources = [line.rstrip('\n') for line in f.readlines()]
            
        predictions = outputs_df.copy()
        predictions = predictions.explode('generations')
        predictions['generations']=predictions['generations'].apply(lambda x: x['text'])
        
        source_predictions_ = pd.DataFrame({'source': sources, 'prediction': predictions['generations'].tolist()}) 
        
    elif task == 'nli':
        sources = pd.read_json(sources_file, lines=True)
        predictions = outputs_df.copy()
        try:
            sources['premise']=sources.prompt.apply(lambda x: x['premise'])
            sources['hypothesis']=sources.prompt.apply(lambda x: x['hypothesis'])
            
            sources['generation']=predictions.generations.apply(lambda x: x[0]['text'])
            source_predictions_ = sources.rename(columns={'hypothesis': 'source', 'generation':'prediction'})
        except:
            sources = sources.explode('generations', ignore_index=True)
            sources['premise']=sources.prompt.apply(lambda x: x['text'])
            sources['source']=sources.generations.apply(lambda x: x['text'])   

            predictions = predictions.explode('generations', ignore_index=True)
            predictions['premise']=predictions.prompt.apply(lambda x: x['text'])
            predictions['prediction'] = predictions['generations'].apply(lambda x: x['text'])        

            source_predictions_ = pd.concat([sources[['source']], predictions[['prediction']]], axis=1)
        

    ## start evaluation
    ## -- BLEU, SBLEU
    # https://huggingface.co/spaces/evaluate-metric/sacrebleu
    sacrebleu = evaluate.load("sacrebleu")
    # decided not to save raw sbleu score since it took a while to compute
    # sbleu_score_raw = [sacrebleu.compute(predictions=[predictions[i]], references=[sources[i]])['score'] for i in range(len(predictions))]
    sbleu_score = sacrebleu.compute(
        predictions=source_predictions_['prediction'].tolist(), references=source_predictions_['source'].tolist()
    )["score"]

    ## -- BERTScore, SBERTScore
    # https://huggingface.co/spaces/evaluate-metric/bertscore
    # The function returns a dictionary with the following keys - precision, recall, f1, hashcode - and corresponding values for each sentence
    bertscore = evaluate.load("bertscore")
    sbert_score_raw = np.array(
        bertscore.compute(
            predictions=source_predictions_['prediction'].tolist(),
            references=source_predictions_['source'].tolist(),
            lang="en",
            rescale_with_baseline=True,
        )["f1"]
    )
    # Take the mean of f1 scores for all the predictions
    sbert_score = np.mean(sbert_score_raw)


    sbertscore_outputs = pd.DataFrame(
        {"sbert_score": sbert_score_raw}
    )
    sbertscore_outputs.to_csv(results_file + ".sbertscore", index=False)

    # Calculate % of outputs with SBERT score >= 0.5
    sbert_preserved_prop = (sbert_score_raw >= 0.5).mean()
    
    # Calculate count of outputs with SBERT score >= 0.5
    sbert_preserved_count = (sbert_score_raw >= 0.5).sum()

    return sbleu_score, sbert_score, sbert_preserved_prop, sbert_preserved_count


def save_qualitative_results(task,
                             source_file_path, 
                             outputs_df, 
                             ppl_results_path, 
                             constraint_results_path, 
                             contents_prsrv_results_path,
                             qual_results_path):
    
    
    # read files

    outputs = outputs_df.copy()
    outputs = unravel(outputs)
    
    ppl = read_metric_file(ppl_results_path, 'ppl-big-qwen')['ppl'].copy()
    
    if (task=='formality'):
        constraint_sat = read_metric_file(constraint_results_path, 'formality_ext')
    elif (task in ['set-consistency', 'set_nli', 'set-nli', 'set_snli', 'set-snli', 'lconvqa', 'set_lconvqa', 'sc_energy', 'set-lconvqa']):
        constraint_sat = 1 - read_metric_file(constraint_results_path, 'set-consistency-gpt')['sc_class_gpt'].copy() # originally, 1 = incon, 0 = con.
    elif (task=='nli'):
        constraint_sat = read_metric_file(constraint_results_path, 'nli')['nli_class'].apply(lambda x: 0 if x == 'contradiction' else 1).copy()
    elif (task=='toxicity'):
        constraint_sat = 1 - read_metric_file(constraint_results_path, 'toxicity')['toxicity'].copy()
    elif (task=='sentiment'):
        constraint_sat = read_metric_file(constraint_results_path, 'sentiment_ext')
    else:
        raise ValueError(f"Unknown task: {task}")
    
    if os.path.exists(contents_prsrv_results_path):
        contents_prsrv=read_metric_file(contents_prsrv_results_path, 'sbertscore')
    else:
        contents_prsrv = np.ones(len(outputs)) * np.nan


    # preprocess files
    ## key (row index), prompt, gen 
    if (task=='formality'):
        with open(source_file_path, 'r') as f:
            source = [_line.rstrip('\n') for _line in f.readlines()]
        source = pd.DataFrame({'prompt': ["" for _ in range(len(source))], 'generations': source})
    else:
        source = pd.read_json(source_file_path, lines=True)
        source = unravel(source)
    
        if (task=='sentiment'):
            source = source[['prompt','generations']].copy()
        
    

    final_df=pd.concat([source,outputs[['generations']],pd.Series(ppl),pd.Series(constraint_sat),pd.Series(contents_prsrv)],axis=1,ignore_index=True)

    final_df.columns=['prompt','original','edited','ppl','constraint_sat','sbert_score']
    final_df.to_excel(qual_results_path,index=False)

def sentiment_classify_gpt4o(generations_df, output_file_path):
    api_key = os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)
    responses_by_prompt = []
    responses_unravel = []
    system_prompt = """\"Classify each of the following text samples as either Positive or Negative based on their sentiment. Do not include a Neutral class, and ensure each sample is distinctly categorized as either Positive or Negative. The number of examples is 20. Ensure you label every example provided.  Provide the output in JSON format as follows: {'results': ['Positive', 'Negative', ...]}.\"
Text Samples:
"""
    # print(f"Number of prompts: {len(generations_df)}")
    for i in range(len(generations_df)):
        
        prompt = generations_df['prompt'][i]['text']
        generations = generations_df['generations'][i]
        
        full_text = [prompt + x['text'] for x in generations]
        # print(f"Number of generations for {i}th prompt: {len(full_text)}")
        formatted_full_text = ""
        for text in full_text:
            formatted_full_text += "'" + text + "'" + ',\n\n'

        response = client.chat.completions.create(model='gpt-4o-2024-08-06', 
                                                temperature = 0, n = 1, max_tokens=200, #logprobs=True, 
                                                response_format={ 'type': "json_object" },
                                                messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": formatted_full_text}
        ])
        
        
        result = json.loads(response.choices[0].message.content)
        result = result['results']
        result = [1 if x == "Positive" else 0 for x in result]
        # print(f"Number of predictions for {i}th prompt: {len(result)}")
        assert len(full_text) == len(result)

        responses_by_prompt.append(result)
        responses_unravel.extend(result)
        
    # responses.append(response.choices[0].message.content)
    
    with open(output_file_path ,'w') as f:
        
        f.writelines([str(x) + '\n' for x in responses_unravel])
        
    responses_unravel = np.array(responses_unravel)
    return np.mean(responses_unravel), np.std(responses_unravel)



        
def set_consistency_score(generations_df, output_file, device, 
                          config_path):

    
    # load model 
    model = load_sc_energy_model(config_path, device)

    # define dataset and dataloader
    generations_df = generations_df.explode('generations')
    generations = generations_df["generations"].tolist()
    # Extract only 'text' field to avoid collation issues with variable-sized fields
    # Each generation dict might have other fields (lists, arrays) of different sizes
    generations_text_only = [{'text': gen['text'] if gen['text'].startswith('<s>') else '<s> ' + gen['text']} for gen in generations]
    
    dataset = Dataset.from_list(generations_text_only)
   
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # calculate set consistency score
    sc_scores = []
    sc_preds = []
    for batch in dataloader:
        with torch.no_grad():
             # set consistency verification
            batch_text = batch['text']
            output = model.energy_model(batch_text, pair_only = True)["predictions"]
            
            if (model.output_form == 'real_num'):
                probs = output.reshape(-1)
            elif (model.output_form == '2dim_vec'):
                probs = F.softmax(output, dim=-1)[:, -1]
            else:
                raise ValueError(f"Unsupported output form: {model.output_form}")
            sc_scores.extend(probs.tolist())
            sc_preds.extend(torch.where(probs <= model.threshold,0,1).tolist())
    
    # write set consistency score and class to output file
    with open(output_file, 'w') as f:
        
        for score, pred in zip(sc_scores, sc_preds):
            f.write(f"{score},{pred}\n")
    
    # return empirical set consistent probability and average set consistent score
    avg_sc_score = np.nanmean(sc_scores)
    contradiction_rate = np.nanmean(sc_preds)
    return avg_sc_score, contradiction_rate
    

def detect_span(set_text, cls_token, sep_token):

    # set_text == text, e.g., '<s> qa pair 1 </s> qa pair 2 ... </s>

    out = set_text[len(cls_token):].split(sep_token)[:-1]
    
    return [o.strip()+sep_token for o in out]
    
def avg_num_instances(generations_df, output_file, cls_token='<s>', sep_token='.'):
    
    generations_df = generations_df.explode('generations')
    generations = generations_df["generations"].tolist()
    num_instances = []
    for generation in generations:
        spans = detect_span(generation['text'], cls_token=cls_token, sep_token=sep_token)
        num_instances.append(len(spans))
    
    with open(output_file, 'w') as f:
        f.writelines([str(x)+'\n' for x in num_instances])
    
    return np.nanmean(num_instances)



def set_consistency_score_gpt(raw_data, model_name,  output_file='', dataset='lconvqa', shot_num=5):

    params = dict(
        dataset=dataset, # one of ['lconvqa', 'set_nli']
        task='prediction',
        baseline=dict(
            type='llm', 
            model=model_name,
            shot_num=shot_num,
            prediction_type='all_in_one',
            reasoning_effort='medium'
        ),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=1,
    )


    raw_texts = raw_data['generations'].apply(lambda x: x[0]['text']).tolist()

    test_dataset = []
    for _text in raw_texts:
        # logger.info(f"_text: {_text}")
        parsed_qa_pairs = parse_set_text(_text, source_mode="ebm", dataset=params['dataset'])
        # logger.info(f"parsed_qa_pairs: {parsed_qa_pairs}")
        test_dataset.append(parsed_qa_pairs)
        
    test_dataset = set_consistency_dataset(test_dataset)
    
    dataloader = lm_loader(test_dataset, params=params).get_loader()
    model = baseline_model(params, 'prediction')
    pred = [] # 0: consistent, 1: inconsistent

    f = open(output_file, 'w') if output_file != '' else None

    fail = False
    for i, pairs in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i == 0:
            print(f"pairs: {pairs}")
        for try_num in range(3): # max 3 tries
            try:
                evaluate_result = model.predict(pairs)
                fail = False
                break
            except Exception as e:
                logger.warning(f"================")
                logger.warning(f"Error in prediction for index {i} during {try_num}th try: {e}")
                logger.warning(f"pairs: {pairs}")
                logger.warning(f"================")
                fail = True
        if fail:
            evaluate_result = {'pred': [torch.nan] * len(pairs)}
                
        pred.extend(evaluate_result['pred'])
        if f is not None:
            f.writelines([str(x)+'\n' for x in evaluate_result['pred']])
            f.flush()

    if f is not None:
        f.close()
    
    contradiction_rate = sum(pred) / len(pred)
    
    return contradiction_rate   

