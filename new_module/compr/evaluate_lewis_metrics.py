import argparse
import logging
import os
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import wandb
from evaluation.prompted_sampling.evaluate import (
    conditional_perplexity,
    distinctness,
    fluency_classify,
    formality_score_ext,
    formality_score_int,
    repetition,
    sentiment_classify_big,
    sentiment_classify_own2,
    toxicity_score,
    toxicity_score_energy,
    toxicity_score_int,
    toxicity_score_mucola,
)

## logging-related
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger("le")
logger.setLevel(logging.DEBUG)


def sentiment_classify_lewis_metrics(predictions):
    classifier = pipeline("sentiment-analysis", device=0)
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')

    sentiment_predictions_all = []
    batch_size = 16
    for i in range(0, len(predictions), batch_size):
        batch = predictions[i : i + batch_size]
        try:
            sentiment_predictions = classifier(batch)
            sentiment_predictions = [
                1 if x["label"] == "POSITIVE" else 0 for x in sentiment_predictions
            ]
        except IndexError:  # sometimes the generation is too long?
            print("exception occured, please check")
            sentiment_predictions = [np.nan] * len(batch)

        sentiment_predictions_all.extend(sentiment_predictions)

    return sentiment_predictions_all


def evaluate_lewis_metrics(args):
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

    with open("data/Sentiment-and-Style-Transfer/data/yelp/sentiment.test.0") as f:
        source_0 = [line.rstrip("\n") for line in f.readlines()]
    with open("data/Sentiment-and-Style-Transfer/data/yelp/sentiment.test.1") as f:
        source_1 = [line.rstrip("\n") for line in f.readlines()]

    ref_0 = pd.read_csv(
        "data/Sentiment-and-Style-Transfer/data/yelp/reference.0",
        sep="\t",
        header=None,
        names=["text", "ref"],
    )
    ref_1 = pd.read_csv(
        "data/Sentiment-and-Style-Transfer/data/yelp/reference.1",
        sep="\t",
        header=None,
        names=["text", "ref"],
    )

    # predictions_0 = pd.read_json('outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-dppnf7pu-negative-to-positive/outputs_epsilon0.75.txt', lines=True)
    # predictions_1 = pd.read_json('outputs/sentiment/mlm-reranking/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/lewis-compr/mlm-beamsearch-v0-word-nps5-k10-beam5-allsat_primary-yv93byiz-positive-to-negative/outputs_epsilon0.75.txt', lines=True)
    try:
        predictions_0 = pd.read_json(args.outputs_file_0, lines=True)
        predictions_1 = pd.read_json(args.outputs_file_1, lines=True)

        ## unravel prediction data
        predictions_0 = predictions_0.explode("generations")
        predictions_0["text"] = predictions_0["generations"].apply(lambda x: x["text"])
        predictions_0 = predictions_0[["text"]].copy()

        predictions_1 = predictions_1.explode("generations")
        predictions_1["text"] = predictions_1["generations"].apply(lambda x: x["text"])
        predictions_1 = predictions_1[["text"]].copy()
        # predictions = predictions_0['text'].tolist() + predictions_1['text'].tolist()
        predictions = predictions_0["text"].tolist()
        ## added to take into account some \n's are trailing the text
        predictions = [x.rstrip("\n") for x in predictions]
    except:
        with open(args.outputs_file_0, "r") as f:
            predictions_0 = [line.rstrip("\n") for line in f.readlines()]
        with open(args.outputs_file_1, "r") as f:
            predictions_1 = [line.rstrip("\n") for line in f.readlines()]

        predictions_0 = [
            tokenizer.decode(tokenizer.encode(x, add_special_tokens=False))
            for x in predictions_0
        ]
        predictions_1 = [
            tokenizer.decode(tokenizer.encode(x, add_special_tokens=False))
            for x in predictions_1
        ]
        # predictions = predictions_0 + predictions_1
        predictions = predictions_0

    ## detokenize source data -> not sure if this is necessary or if this is the right way
    source_0 = [
        tokenizer.decode(tokenizer.encode(x, add_special_tokens=False))
        for x in source_0
    ]
    source_1 = [
        tokenizer.decode(tokenizer.encode(x, add_special_tokens=False))
        for x in source_1
    ]
    # sources = source_0 + source_1
    sources = source_0

    ## detokenize reference data -> not sure if this is necessary or if this is the right way
    ref_0["ref"] = ref_0["ref"].apply(
        lambda x: tokenizer.decode(tokenizer.encode(x, add_special_tokens=False))
    )
    ref_1["ref"] = ref_1["ref"].apply(
        lambda x: tokenizer.decode(tokenizer.encode(x, add_special_tokens=False))
    )
    # references = ref_0['ref'].tolist() + ref_1['ref'].tolist()
    references = ref_0["ref"].tolist()

    # https://huggingface.co/spaces/evaluate-metric/sacrebleu
    sacrebleu = evaluate.load("sacrebleu")
    # bleu_score_raw = [sacrebleu.compute(predictions=[predictions[i]], references=[references[i]])['score'] for i in range(len(predictions))]
    bleu_score = sacrebleu.compute(
        predictions=predictions, references=[[sent] for sent in references]
    )["score"]
    # print(np.mean(bleu_score_raw), bleu_score)
    # sbleu_score_raw = [sacrebleu.compute(predictions=[predictions[i]], references=[sources[i]])['score'] for i in range(len(predictions))]
    sbleu_score = sacrebleu.compute(
        predictions=predictions, references=[[sent] for sent in sources]
    )["score"]
    # print(np.mean(sbleu_score_raw), sbleu_score)

    # https://huggingface.co/spaces/evaluate-metric/bertscore
    # The function returns a dictionary with the following keys - precision, recall, f1, hashcode - and corresponding values for each sentence
    # Take the mean of f1 scores for all the predictions
    bertscore = evaluate.load("bertscore")
    bert_score_raw = np.array(
        bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            rescale_with_baseline=True,
        )["f1"]
    )
    sbert_score_raw = np.array(
        bertscore.compute(
            predictions=predictions,
            references=sources,
            lang="en",
            rescale_with_baseline=True,
        )["f1"]
    )
    bert_score = np.mean(bert_score_raw)
    sbert_score = np.mean(sbert_score_raw)

    # since predictions_{class} are the results of sentiment transfer from the class to the opposite class,
    # labels should be 1 for predictions_0 and 0 for predictions_1
    sentiment_predictions = sentiment_classify_lewis_metrics(predictions)
    # sentiment_labels = [1] * len(predictions_0) + [0] * len(predictions_1)
    sentiment_labels = [1] * len(predictions)
    acc_score = accuracy_score(sentiment_labels, sentiment_predictions)

    if args.update_wandb:
        api = wandb.Api()
        run = api.run(args.wandb_run_path)
        run.summary.update(
            {
                "acc": [acc_score * 100],
                "sbleu": [sbleu_score],
                "bleu": [bleu_score],
                "sbert": [sbert_score * 100],
                "bert": [bert_score * 100],
            }
        )
        run.update()
    elif args.results_file is not None:
        metrics = pd.DataFrame(
            {
                "acc": [acc_score * 100],
                "sbleu": [sbleu_score],
                "bleu": [bleu_score],
                "sbert": [sbert_score * 100],
                "bert": [bert_score * 100],
            }
        )
        metrics.to_csv(args.results_file, index=False)

        classify_outputs = pd.DataFrame(
            {
                "sentiment_predictions": sentiment_predictions,
                "sentiment_labels": sentiment_labels,
            }
        )
        classify_outputs.to_csv(args.results_file + ".cls", index=False)

        bertscore_outputs = pd.DataFrame(
            {"bert_score": bert_score_raw, "sbert_score": sbert_score_raw}
        )
        bertscore_outputs.to_csv(args.results_file + ".bertscore", index=False)

    return acc_score, bleu_score, sbleu_score, bert_score, sbert_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--update_wandb",
        action="store_true",
        help="Whether to update metrics to wandb.",
    )
    parser.add_argument(
        "--wandb_run_path", type=str, help="Wandb run path of a previous run to eval."
    )
    parser.add_argument(
        "--outputs_file_0",
        type=str,
        help="Path to the outputs file for negative to positive transfer.",
    )
    parser.add_argument(
        "--outputs_file_1",
        type=str,
        help="Path to the outputs file for positive to negative transfer.",
    )
    parser.add_argument(
        "--results_file", type=str, help="Path to the results file to save the metrics."
    )

    args = parser.parse_args()

    evaluate_lewis_metrics(args)
