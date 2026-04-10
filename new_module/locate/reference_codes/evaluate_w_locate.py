import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, TextClassificationPipeline

import argparse
import json
import os
import operator

from functools import partial
from collections import Counter
from scipy import stats
from multiprocessing.pool import Pool

import random

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# from perspective_api import PerspectiveWorker, unpack_scores

from transformers import GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, SequenceClassifierOutput

from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

import torch.nn as nn
import torch

import logging

import gc

logger = logging.getLogger(__name__)

    
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


# def recall(out, labels, k=4):
#     idx_array = stats.rankdata(-out, axis=1, method='min')
#     labels = labels.astype(int)
#     rank = np.take_along_axis(idx_array, labels[:,None], axis=1)
#     return np.count_nonzero(rank<=k)

# def calculate_recall(_docs_with_score, _labels):
#     _docs = [x[0].metadata['application_number'] for x in _docs_with_score] # list of application numbers with length same as docs_with_score
#     _yn = [1 if x in _labels else 0 for x in _docs] # list of 1,0 with length same as docs_with_score
#     _scores = np.array([x[1] for x in _docs_with_score]) # array of shape (length docs_with_score,) of similarity scores
    
#     idx_array = stats.rankdata(_scores, method='max')
#     idx_array_gold = idx_array[np.where(np.array(_yn)==1)[0]] # only get ranks of indices where application number is in the labels (gold prior arts)
#     if len(idx_array_gold) == 0:
#         return 0.
#     else:
#         return len(idx_array_gold) / len(_docs_with_score)

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


@click.command()
@click.option('--generations_file', required=True, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--output_file', required=True, type=str, help='filename to write outputs')
@click.option('--metrics', required=True, type=str, help='which metrics to compute, write comma separeted, ppl-own,ppl-big,cola,self-bleu,zipf,repetition,dist-n,sentiment')
@click.option('--extra', required=False, type=str, help='extra params like which topic category or keyword file')
@click.option('--intermediate_file', required=True, type=str, help='a jsonl file with intermediate generations and locate indices')
@click.option('--label_file', required=True, type=str, help='a jsonl file with locate labels')
def main(generations_file, output_file, metrics, extra):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    if generations_file.endswith(".jsonl"):
        generations_df = pd.read_json(generations_file, lines=True)
    else:
        # with open(generations_file) as fin:
        #     generations_df = [{'prompt':{'text':''}, 'generations':[{'text':l.strip()}]} for l in fin.readlines()]
        #     generations_df = pd.DataFrame(generations_df)
        
        # (23-03-24: hyeryung) ^ above code results in empty prompt column. it results in the following error: 
        # RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
        generations_df = pd.read_json(generations_file, lines=True) 
        print(generations_df.head())
    label_df = pd.read_json(label_file, line=True)
    intermediate_df = pd.read_json(intermediate_file, line=True)
    
    
    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    # Fluency
    fo = open(output_dir / output_file, 'w') #just creating the file
    fo.close()
    if "pplcustom" in metrics:
        print("ppl")
        allmetrics = metrics.split(",")
        for metric in allmetrics:
            if "pplcustom" in metric:
                eval_modelname = metric.split("#")[1]
                print(eval_modelname)
                eval_model = AutoModelForCausalLM.from_pretrained(eval_modelname).to(device)
                eval_tokenizer = AutoTokenizer.from_pretrained(eval_modelname)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-"+eval_modelname.replace("/", "-")))

                # write output results
                with open(output_dir / output_file, 'a') as fo:
                    fo.write(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')
                    print(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')

    if "ppl-big" in metricset: #GPT2-XL
        print("big")
        
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2-xl perplexity, gpt2-xl total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2-xl perplexity, gpt2-xl total perplexity = {ppl}, {total_ppl}\n')

    
    if "ppl-own" in metricset: #GPT2-Large
        print("own")
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-own"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2-large perplexity, gpt2-large total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2-large perplexity, gpt2-large total perplexity = {ppl}, {total_ppl}\n')
        
    if "ppl-small" in metricset: #GPT2
        print("small")
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-own"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
    
    if 'sentiment' in metricset:
        print("sentiment-big") #c3
        sentiment_accuracy, sentiment_std = sentiment_classify_big(generations_df, sentiment_file=output_dir / (output_file+".sentiment-big"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment-big accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment-big accuracy = {sentiment_accuracy}, {sentiment_std}')

        print("sentiment") #c1
        sentiment_accuracy, sentiment_std = sentiment_classify(generations_df, sentiment_file=output_dir / (output_file+".sentiment"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}')

        print("sentiment-yelp") #c2
        sentiment_accuracy, sentiment_std = sentiment_classify_yelp(generations_df, sentiment_file=output_dir / (output_file+".sentiment-yelp"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment-yelp accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment-yelp accuracy = {sentiment_accuracy}, {sentiment_std}')

        print("sentiment-own") #internal classifier
        sentiment_accuracy, sentiment_std = sentiment_classify_own2(generations_df, sentiment_file=output_dir / (output_file+".sentiment-own"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment-own accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment-own accuracy = {sentiment_accuracy}, {sentiment_std}')
    
    if 'sentiment_promptonly' in metricset:
        print("sentiment")
        sentiment_accuracy, sentiment_std, num_neutral = sentiment_classify_promptonly(generations_df, sentiment_file=output_dir / (output_file+".sentiment"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}, {num_neutral}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}, {num_neutral}')
    
    if 'toxicity' in metricset:
        print("toxicity")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score(generations_df, perspective_file=output_dir / (output_file+".toxicity"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'[perspective] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
            print(f'[perspective] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
            
    if 'toxicity-energy' in metricset:
        print("toxicity-energy")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_energy(generations_df, toxicity_file=output_dir / (output_file+".toxicity_energy"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'[energy model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
            print(f'[energy model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
            
            
    if 'toxicity-mucola' in metricset:
        print("toxicity-mucola")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_mucola(generations_df, toxicity_file=output_dir / (output_file+".toxicity_mucola"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'[mucola model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
            print(f'[mucola model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
    
    #cola
    if "cola" in metricset:
        cola_accuracy = fluency_classify(generations_df, output_file=output_dir / (output_file+".cola"))
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'cola acceptability accuracy = {cola_accuracy}\n')
            print(cola_accuracy)

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    # self-bleu
    if "self-bleu" in metricset:
        bleus = self_bleu(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+1} = {bleu}\n')
                print(f'bleu-{i+1} = {bleu}')
    
    if "dist-n2" in metricset:
        dist1, dist2, dist3 = distinctness2(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    # self-bleu
    if "self-bleu2" in metricset:
        bleus = self_bleu2(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+3} = {bleu}\n')
                print(f'bleu-{i+3} = {bleu}')

    # zipf-coefficient
    if "zipf" in metricset:
        s, r, p = zipf_coefficient(generations_df)
        
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'zipf: s={s}, r={r}, p={p}\n')
            print(f'zipf: s={s}, r={r}, p={p}')

    # repetition
    if "repetition" in metricset:
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        rep_rate = repetition(generations_df, eval_tokenizer, rep_file=output_dir / (output_file+".repetitions"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'repetition_rate: {rep_rate}')
            print(f'repetition_rate: {rep_rate}')

    if "allsat" in metricset:
        print("allsat")
        sat_accuracy, sat_std, sat_once = allsat_accuracy(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean allsat accuracy, std = {sat_accuracy}--{sat_std}, {sat_once}\n')
            print(f'mean allsat accuracy, std = {sat_accuracy}--{sat_std}, {sat_once}')
    
    if "keywordcount" in metricset:
        print("keywordcount")
        print(extra)
        bestcount, allcount = keyword_count_coverage(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean allsat accuracy, std = {bestcount}--{allcount}\n')
            print(f'mean allsat accuracy, std = {bestcount}--{allcount}')
    
    if "politeness" in metricset:
        print("politeness")
        polit_accuracy, politeness_std, politeness_once = politeness_classify(generations_df, politeness_file=output_dir / (output_file+".politeness"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean politeness accuracy, std = {polit_accuracy}--{politeness_std}, {politeness_once}\n')
            print(f'mean politeness accuracy, std = {polit_accuracy}--{politeness_std}, {politeness_once}')
    
    if "topic" in metricset:
        print(f"topic -- {extra}")
        num_match, num_unit_match, total_sent_count = topic_eval(generations_df, extra, None)

        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'num match, num unit match, total sent count: {num_match}--{num_unit_match}, {total_sent_count}\n')
            print(f'num match, num unit match, total sent count: {num_match}--{num_unit_match}, {total_sent_count}')

    # HUSE: TODO
    
    if "locate_precision" in metricset:
        print("locate_precision")
        precision_label_1_if_gt0, precision_label_1_if_eq1 = locate_precision(intermediate_df, label_df)

        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'locate precision (if label is 1 if >0): {precision_label_1_if_gt0}\n')
            fo.write(f'locate precision (if label is 1 if ==1): {precision_label_1_if_eq1}\n')
            print(f'locate precision (if label is 1 if >0): {precision_label_1_if_gt0}\n')
            print(f'locate precision (if label is 1 if ==1): {precision_label_1_if_eq1}\n')
    
    if "locate_recall" in metricset:
        print("locate_recall")
        recall_label_1_if_gt0, recall_label_1_if_eq1 = locate_recall(intermediate_df, label_df)

        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'locate recall (if label is 1 if >0): {recall_label_1_if_gt0}\n')
            fo.write(f'locate recall (if label is 1 if ==1): {recall_label_1_if_eq1}\n')
            print(f'locate recall (if label is 1 if >0): {recall_label_1_if_gt0}\n')
            print(f'locate recall (if label is 1 if ==1): {recall_label_1_if_eq1}\n')

if __name__ == '__main__':
#     generations_file = '/home/hyeryung_son/mucoco/outputs/toxicity/locate-edit-jigsaw-loc-alltoks--1steps-project-1steps-mrr_allsat/outputs_epsilon-3.txt'
#     output_file = f'{generations_file}.metrics'
#     # metrics = 'toxicity-mucola,toxicity-energy'
#     metrics = 'toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n'
#     extra = None
    # main(generations_file, output_file, metrics, extra)
    main()