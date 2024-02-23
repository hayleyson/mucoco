import argparse
import logging
import os
import sys
from pathlib import Path

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

def evaluate(run_path, generations_file_path, metrics, **kwargs):
    """
    kwargs: 
    - includes "formality_model_path", "formality_model_type" for formality-int score
    - includes "sentiment_model_path", "sentiment_model_type" for sentiment-int score
    - includes "toxicity_model_path", "toxicity_model_type" for toxicity-int score
    """
      
    generations_df = pd.read_json(generations_file_path, lines=True) 
    logger.debug(generations_df.shape)

    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(metricset)
    
    if run_path != "": ## if wandb run path is provided.
        api = wandb.Api()
        run = api.run(run_path)
        min_epsilon = run.config['min_epsilons'][0] if isinstance(run.config['min_epsilons'],list) else run.config['min_epsilons']
        output_file = f"results_epsilon{min_epsilon}-test.txt"
        
        if run.config.get('task', None) is not None:
            task = run.config['task']
        else:
            task = run.config['lossabbr'].split(':')[1]
            
        if run.config.get('model_paths', None) is not None:
            model_path = run.config['model_paths'][1]
        else:
            model_path = run.config['model'].split(':')[1]
        
        if run.state != 'finished':
            try:
                if task == 'toxicity':
                    assert len(generations_df) == 250
                elif task == 'formality':
                    assert len(generations_df) == 1416
                elif task == 'sentiment':
                    assert len(generations_df) == 15
            except:
                raise Exception(f"The number of generations is not correct. {len(generations_df)} while task is {task}")
            ## if the run state is not finished but the number of generations are complete -> finish the run
            run1 = wandb.init(project=run_path.split('/')[1], id=run_path.split('/')[-1], resume="must")
            run1.finish()
            del run1
        ## update model_tag if it is not set
        model_tag = run.config.get('model_tag', None)
        if (model_tag is None) or (model_tag == ''):
            run.config['model_tag'] = 'em' if ('energy-training' in model_path) else 'clsf'
            if (task == 'formality') and ('gyafc' in model_path):
                run.config['model_tag'] += '-gyafc'
    else:
        output_file = "results.txt"
        
    output_dir = Path(os.path.dirname(generations_file_path))
    if os.path.exists(output_dir / output_file):
        fp = open(output_dir / output_file, 'a')
        fp.write('-'*50+'\n')
    else:
        fp = open(output_dir / output_file, 'w')

    if "ppl-big" in metricset: #GPT2-XL
        logger.debug("big")
        
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big"))
        if run_path != "":
            run.summary.update({'ppl': ppl, 'total_ppl': total_ppl})
        fp.write(f'ppl: {ppl}, total_ppl: {total_ppl}\n')

    if 'toxicity' in metricset:
        logger.debug("toxicity-external")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score(generations_df, perspective_file=output_dir / (output_file+".toxicity"))
        if run_path != "":
            run.summary.update({'avg_max_toxicity': avg_max_toxicity, 'toxic_probability_p': toxic_probability_p,
                'avg_toxicity': avg_toxicity, 'toxic_probability_s': toxic_probability_s})
        fp.write(f'avg_max_toxicity: {avg_max_toxicity}, toxic_probability_p: {toxic_probability_p}, avg_toxicity: {avg_toxicity}, toxic_probability_s: {toxic_probability_s}\n')
            
    # if 'toxicity-energy' in metricset:
    #     logger.debug("toxicity-energy")
    #     (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_energy(generations_df, toxicity_file=output_dir / (output_file+".toxicity_energy"))
    #     if run_path != "":
    #         run.summary.update({'avg_max_toxicity_energy': avg_max_toxicity, 'toxic_probability_p_energy': toxic_probability_p,
    #             'avg_toxicity_energy': avg_toxicity, 'toxic_probability_s_energy': toxic_probability_s})
    #     fp.write(f'avg_max_toxicity_energy: {avg_max_toxicity}, toxic_probability_p_energy: {toxic_probability_p}, avg_toxicity_energy: {avg_toxicity}, toxic_probability_s_energy: {toxic_probability_s}\n')
        
    # if 'toxicity-mucola' in metricset:
    #     logger.debug("toxicity-mucola")
    #     (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_mucola(generations_df, toxicity_file=output_dir / (output_file+".toxicity_mucola"))
    #     if run_path != "":
    #         run.summary.update({'avg_max_toxicity_mucola': avg_max_toxicity, 'toxic_probability_p_mucola': toxic_probability_p,
    #             'avg_toxicity_mucola': avg_toxicity, 'toxic_probability_s_mucola': toxic_probability_s})
    #     fp.write(f'avg_max_toxicity_mucola: {avg_max_toxicity}, toxic_probability_p_mucola: {toxic_probability_p}, avg_toxicity_mucola: {avg_toxicity}, toxic_probability_s_mucola: {toxic_probability_s}\n')
        
    if 'toxicity-int' in metricset:
        logger.debug("toxicity-internal")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_int(generations_df, output_dir / (output_file+".toxicity_int"), device,
                                                                                                        kwargs['toxicity_model_path'], kwargs['toxicity_model_type'])
        if run_path != "":
            run.summary.update({'avg_max_toxicity_int': avg_max_toxicity, 'toxic_probability_p_int': toxic_probability_p,
                'avg_toxicity_int': avg_toxicity, 'toxic_probability_s_int': toxic_probability_s})
        fp.write(f'avg_max_toxicity_int: {avg_max_toxicity}, toxic_probability_p_int: {toxic_probability_p}, avg_toxicity_int: {avg_toxicity}, toxic_probability_s_int: {toxic_probability_s}\n')

    if 'formality-ext' in metricset:
        logger.debug("formality-external")
        avg_formality, formal_proba = formality_score_ext(generations_df, output_dir / (output_file+".formality_ext"), device)
        if run_path != "":
            run.summary.update({'avg_formality': avg_formality, 'formal_proba': formal_proba})
        fp.write(f'avg_formality: {avg_formality}, formal_proba: {formal_proba}\n')
        
    if 'formality-int' in metricset:
        logger.debug("formality-internal")
        avg_formality, formal_proba = formality_score_int(generations_df, output_dir / (output_file+".formality_int"), device, 
                                                          kwargs['formality_model_path'], kwargs['formality_model_type'])
        if run_path != "":
            run.summary.update({'avg_formality_int': avg_formality, 'formal_proba_int': formal_proba})
        fp.write(f'avg_formality_int: {avg_formality}, formal_proba_int: {formal_proba}\n')
        
    if 'sentiment-ext' in metricset:
        logger.debug("sentiment-external")
        positive_proba, std_positive_proba_p, avg_positivity = sentiment_classify_big(generations_df, output_dir / (output_file+".sentiment_ext"))
        if run_path != "":
            # run.summary.update({"avg_sentiment": None, "positive_proba": None, 
            #                 "avg_positive_proba_p": None, "std_positive_proba_p": None, 
            #                 "positive_proba_p_avg": None, "positive_proba_s": None})
            run.summary.update({'positive_proba': positive_proba, 
                                'positive_proba_p_std': std_positive_proba_p,
                                'avg_positivity': avg_positivity})
        fp.write(f'positive_proba: {positive_proba}, positive_proba_p_std: {std_positive_proba_p}, avg_positivity: {avg_positivity}\n')
        
    if 'sentiment-int' in metricset:
        logger.debug("sentiment-internal")
        positive_proba, std_positive_proba_p, avg_positivity = sentiment_classify_own2(generations_df, output_dir / (output_file+".sentiment_int"),
                                                          kwargs['sentiment_model_path'], kwargs['sentiment_model_type'])
        if run_path != "":
            # run.summary.update({"avg_sentiment_int": None, "positive_proba_int": None, 
            #                     "avg_positive_proba_p_int": None, "std_positive_proba_p_int": None, 
            #                     "positive_proba_p_avg_int": None, "positive_proba_s_int": None})
            run.summary.update({'positive_proba_int': positive_proba, 
                                'positive_proba_p_std_int': std_positive_proba_p,
                                'avg_positivity_int': avg_positivity})
        fp.write(f'positive_proba_int: {positive_proba}, positive_proba_p_std_int: {std_positive_proba_p}, avg_positivity_int: {avg_positivity}\n')

    if "dist-n" in metricset:
        logger.debug("dist-n")
        dist1, dist2, dist3 = distinctness(generations_df)
        if run_path != "":
            run.summary.update({'dist-1': dist1, 'dist-2': dist2, 'dist-3': dist3})
        fp.write(f'dist-1: {dist1}, dist-2: {dist2}, dist-3: {dist3}\n')
        
    if "repetition" in metricset:
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        rep_rate = repetition(generations_df, eval_tokenizer, rep_file=output_dir / (output_file+".repetitions"))
        if run_path != "":
            run.summary.update({'rep_rate': rep_rate})
        fp.write(f'repetition_rate: {rep_rate}\n')
        
    if "fluency" in metricset:
        fluency = fluency_classify(generations_df, output_dir / (output_file+".fluency"))
        if run_path != "":
            run.summary.update({'fluent_proba': fluency})
        fp.write(f'fluent_proba: {fluency}\n')
        
    if run_path != "":
        run.update()
    fp.close()        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_path', help='Wandb run path of the decoding run to eval.')
    parser.add_argument('--generations_file_path', help='Path to the decoding outputs file for eval.')
    parser.add_argument('--metrics', default='toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n', help='comma-separated string of a list of metrics for eval.')
    parser.add_argument('--sentiment_model_path', type=str, help='path to sentiment energy model (or binary classifier) used for decoding')
    parser.add_argument('--sentiment_model_type', type=str, choices=['RobertaCustomForSequenceClassification', 'AutoModelForSequenceClassification'], help='the type of sentiment energy model (or binary classifier) used for decoding')
    parser.add_argument('--formality_model_path', type=str, help='path to formality energy model (or binary classifier) used for decoding')
    parser.add_argument('--formality_model_type', type=str, choices=['RobertaCustomForSequenceClassification', 'AutoModelForSequenceClassification'], help='the type of formality energy model (or binary classifier) used for decoding')
    parser.add_argument('--toxicity_model_path', type=str, help='path to toxicity energy model (or binary classifier) used for decoding')
    parser.add_argument('--toxicity_model_type', type=str, choices=['RobertaCustomForSequenceClassification', 'AutoModelForSequenceClassification'], help='the type of toxicity energy model (or binary classifier) used for decoding')

    
    args = parser.parse_args()
    
    
    evaluate(args.generations_file_path, args.metrics, args.run_path, 
             sentiment_model_path=args.sentiment_model_path, sentiment_model_type=args.sentiment_model_type,
             formality_model_path=args.formality_model_path, formality_model_type=args.formality_model_type,
             toxicity_model_path=args.toxicity_model_path, toxicity_model_type=args.toxicity_model_type)