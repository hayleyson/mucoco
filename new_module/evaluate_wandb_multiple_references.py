import argparse
import logging
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("le")
logger.setLevel(logging.DEBUG)


def contents_preservation_metrics(results_file,target_style):
    
    if target_style=='formal':
        
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal','r') as f:
            sources = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal.ref0', 'r') as f:
            ref0 = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal.ref1', 'r') as f:
            ref1 = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal.ref2', 'r') as f:
            ref2 = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal.ref3', 'r') as f:
            ref3 = [line.rstrip('\n') for line in f.readlines()]    
            
    elif target_style=='informal':
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal','r') as f:
            sources = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.ref0', 'r') as f:
            ref0 = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.ref1', 'r') as f:
            ref1 = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.ref2', 'r') as f:
            ref2 = [line.rstrip('\n') for line in f.readlines()]
        with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.ref3', 'r') as f:
            ref3 = [line.rstrip('\n') for line in f.readlines()]                        
    
    predictions = list(zip(ref0, ref1, ref2, ref3))
        
    ## start evaluation
    ## -- BLEU, SBLEU
    # https://huggingface.co/spaces/evaluate-metric/sacrebleu
    sacrebleu = evaluate.load("sacrebleu")
    # decided not to save raw sbleu score since it took a while to compute
    # sbleu_score_raw = [sacrebleu.compute(predictions=[predictions[i]], references=[sources[i]])['score'] for i in range(len(predictions))]
    sbleu_score = sacrebleu.compute(
        predictions=sources, references=predictions
    )["score"]

    ## -- BERTScore, SBERTScore
    # https://huggingface.co/spaces/evaluate-metric/bertscore
    # The function returns a dictionary with the following keys - precision, recall, f1, hashcode - and corresponding values for each sentence
    bertscore = evaluate.load("bertscore")
    sbert_score_raw = np.array(
        bertscore.compute(
            predictions=sources,
            references=predictions,
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

    return sbleu_score, sbert_score*100, sbert_preserved_prop, sbert_preserved_count


def simplified_evaluate_main(generations_file_path, target_style):
    
    
    output_dir = Path(os.path.dirname(generations_file_path))
    output_file = f"{os.path.splitext(generations_file_path.split('/')[-1])[0]}.merged-results.txt"
    print(output_dir / output_file)
    if os.path.exists(output_dir / output_file):
        fp = open(output_dir / output_file, 'a')
        fp.write('-'*50+'\n')
    else:
        fp = open(output_dir / output_file, 'w')
    sbleu_score, sbert_score, sbert_preserved_prop, sbert_preserved_count = contents_preservation_metrics(generations_file_path,target_style)
    fp.write(f"sbleu: {sbleu_score}\n")
    fp.write(f"sbert_score: {sbert_score}, sbert_preserved_prop: {sbert_preserved_prop}, sbert_preserved_count: {sbert_preserved_count}\n")
    fp.close()
    
if __name__ == "__main__":
        
    generations_file_path = '/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal.ref.merged'
    simplified_evaluate_main(generations_file_path, target_style='formal')
    
    generations_file_path = '/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.ref.merged'
    simplified_evaluate_main(generations_file_path, target_style='informal')