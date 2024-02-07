import logging
import os
import argparse
import sys



import wandb
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from evaluation.prompted_sampling.evaluate import conditional_perplexity, toxicity_score, toxicity_score_energy, toxicity_score_mucola, toxicity_score_int, \
    formality_score_int, formality_score_ext, distinctness, repetition, sentiment_classify_own2, sentiment_classify_big, fluency_classify

## logging-related
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("le")
logger.setLevel(logging.DEBUG)

def evaluate(run_path, outfile, metrics, **kwargs):
    """
    kwargs: 
    - includes "formality_model_path", "formality_model_type" for formality-int score
    - includes "sentiment_model_path", "sentiment_model_type" for sentiment-int score
    """
      
    # wandb.init(project="mucola", id=run_path.split('/')[-1], resume="allow")

    api = wandb.Api()
    run = api.run(run_path)

    output_dir = Path(os.path.dirname(outfile))
    output_file = f"results_epsilon{run.config['min_epsilons'][0]}-test.txt"
    generations_df = pd.read_json(outfile, lines=True) 
    logger.debug(generations_df.shape)
    
    if run.state != 'finished':
        try:
            if run.config['task'] == 'toxicity':
                assert len(generations_df) == 250
            elif run.config['task'] == 'formality':
                assert len(generations_df) == 1416
            elif run.config['task'] == 'sentiment':
                assert len(generations_df) == 15
        except:
            # logger.debug(f"len(generations_df) = {len(generations_df)}")
            # logger.debug(f"run.config['task'] = {run.config['task']}")
            raise Exception(f"The number of generations is not correct. {len(generations_df)} while task is {run.config['task']}")
        ## if the generations are complete -> finish the run
        run1 = wandb.init(project=run_path.split('/')[1], id=run_path.split('/')[-1], resume="must")
        run1.finish()
        del run1

    model_tag = run.config.get('model_tag', None)
    if (model_tag is None) or (model_tag == ''):
        run.config['model_tag'] = 'em' if ('energy-training' in run.config['model_paths'][1]) else 'clsf'
        if (run.config['task'] == 'formality') and ('gyafc' in run.config['model_paths'][1]):
            run.config['model_tag'] += '-gyafc'
            

    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(metricset)

    logger.debug(str(run.summary))

    if "ppl-big" in metricset: #GPT2-XL
        logger.debug("big")
        
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big"))
        run.summary.update({'ppl': ppl, 'total_ppl': total_ppl})

    if 'toxicity' in metricset:
        logger.debug("toxicity")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score(generations_df, perspective_file=output_dir / (output_file+".toxicity"))
        run.summary.update({'avg_max_toxicity': avg_max_toxicity, 'toxic_probability_p': toxic_probability_p,
                'avg_toxicity': avg_toxicity, 'toxic_probability_s': toxic_probability_s})
            
    if 'toxicity-energy' in metricset:
        logger.debug("toxicity-energy")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_energy(generations_df, toxicity_file=output_dir / (output_file+".toxicity_energy"))
        run.summary.update({'avg_max_toxicity_energy': avg_max_toxicity, 'toxic_probability_p_energy': toxic_probability_p,
                'avg_toxicity_energy': avg_toxicity, 'toxic_probability_s_energy': toxic_probability_s})
        # with open(output_dir / output_file, 'a') as fo:
        #     fo.write(f'[energy model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
        #     logger.debug(f'[energy model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
            
    if 'toxicity-mucola' in metricset:
        logger.debug("toxicity-mucola")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_mucola(generations_df, toxicity_file=output_dir / (output_file+".toxicity_mucola"))
        run.summary.update({'avg_max_toxicity_mucola': avg_max_toxicity, 'toxic_probability_p_mucola': toxic_probability_p,
                'avg_toxicity_mucola': avg_toxicity, 'toxic_probability_s_mucola': toxic_probability_s})
        # with open(output_dir / output_file, 'a') as fo:
        #     fo.write(f'[mucola model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
        #     logger.debug(f'[mucola model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')

    if 'toxicity-int' in metricset:
        logger.debug("toxicity-internal")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_int(generations_df, output_dir / (output_file+".toxicity_int"), device,
                                                                                                        kwargs['toxicity_model_path'], kwargs['toxicity_model_type'])
        run.summary.update({'avg_max_toxicity_int': avg_max_toxicity, 'toxic_probability_p_int': toxic_probability_p,
                'avg_toxicity_int': avg_toxicity, 'toxic_probability_s_int': toxic_probability_s})

    if 'formality-ext' in metricset:
        logger.debug("formality-external")
        avg_formality, formal_proba = formality_score_ext(generations_df, output_dir / (output_file+".formality_ext"), device)
        run.summary.update({'avg_formality': avg_formality, 'formal_proba': formal_proba})
        
    if 'formality-int' in metricset:
        logger.debug("formality-internal")
        avg_formality, formal_proba = formality_score_int(generations_df, output_dir / (output_file+".formality_int"), device, 
                                                          kwargs['formality_model_path'], kwargs['formality_model_type'])
        run.summary.update({'avg_formality_int': avg_formality, 'formal_proba_int': formal_proba})
        
    if 'sentiment-ext' in metricset:
        logger.debug("sentiment-external")
        positive_proba, std_positive_proba_p, avg_positivity = sentiment_classify_big(generations_df, output_dir / (output_file+".sentiment_ext"))
        run.summary.update({"avg_sentiment": None, "positive_proba": None, 
                            "avg_positive_proba_p": None, "std_positive_proba_p": None, 
                            "positive_proba": None, "positive_proba_p_avg": None, "positive_proba_s": None})
        # run.summary.pop('avg_sentiment', None) ## run 하고 나서 지우기 
        # run.summary.pop('positive_proba', None) ## run 하고 나서 지우기 
        run.summary.update({'positive_proba': positive_proba, 
                            'positive_proba_p_std': std_positive_proba_p,
                            'avg_positivity': avg_positivity})
        
    if 'sentiment-int' in metricset:
        logger.debug("sentiment-internal")
        positive_proba, std_positive_proba_p, avg_positivity = sentiment_classify_own2(generations_df, output_dir / (output_file+".sentiment_int"),
                                                          kwargs['sentiment_model_path'], kwargs['sentiment_model_type'])
        run.summary.update({"avg_sentiment_int": None, "positive_proba_int": None, 
                            "avg_positive_proba_p_int": None, "std_positive_proba_p_int": None, 
                            "positive_proba_int": None, "positive_proba_p_avg_int": None, "positive_proba_s_int": None})
        # run.summary.pop('avg_sentiment_int', None) ## run 하고 나서 지우기 
        # run.summary.pop('positive_proba_int',None) ## run 하고 나서 지우기 
        run.summary.update({'positive_proba_int': positive_proba, 
                            'positive_proba_p_std_int': std_positive_proba_p,
                            'avg_positivity_int': avg_positivity})

    if "dist-n" in metricset:
        logger.debug("dist-n")
        dist1, dist2, dist3 = distinctness(generations_df)
        run.summary.update({'dist-1': dist1, 'dist-2': dist2, 'dist-3': dist3})
        # # write output results
        # with open(output_dir / output_file, 'a') as fo:
        #     for i, dist_n in enumerate([dist1, dist2, dist3]):
        #         fo.write(f'dist-{i+1} = {dist_n}\n')
        #         print(f'dist-{i+1} = {dist_n}')
        
        # repetition
    if "repetition" in metricset:
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        rep_rate = repetition(generations_df, eval_tokenizer, rep_file=output_dir / (output_file+".repetitions"))
        run.summary.update({'rep_rate': rep_rate})
        # with open(output_dir / output_file, 'a') as fo:
        #     fo.write(f'repetition_rate: {rep_rate}')
        #     print(f'repetition_rate: {rep_rate}')
        
    if "fluency" in metricset:
        fluency = fluency_classify(generations_df, output_dir / (output_file+".fluency"))
        run.summary.update({'fluent_proba': fluency})
        
    run.update()
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_path', help='Run path of a previous run to eval.')
    parser.add_argument('--outfile', help='Path to the outputs file for eval.')
    parser.add_argument('--metrics', default='toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n', help='comma-separated string of a list of metrics for eval.')
    
    args = parser.parse_args()
    
    evaluate(args.run_path, args.outfile, args.metrics)