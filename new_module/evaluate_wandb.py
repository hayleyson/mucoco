import logging
import os
import argparse
import sys
sys.path.append("/home/s3/hyeryung/mucoco")

import wandb
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from evaluation.prompted_sampling.evaluate import conditional_perplexity, toxicity_score, toxicity_score_energy, toxicity_score_mucola, distinctness

## logging-related
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("le")
logger.setLevel(logging.DEBUG)

def main(run_path, outfile, metrics):
    
    wandb.init(project="mucola", id=run_path.split('/')[-1], resume="must")
    
    api = wandb.Api()
    run = api.run(run_path)

    output_dir = Path(os.path.dirname(outfile))
    output_file = f"results_epsilon{run.config['min_epsilons'][0]}-test.txt"
    generations_df = pd.read_json(outfile, lines=True) 
    logger.debug(generations_df.shape)

    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(metricset)

    if "ppl-big" in metricset: #GPT2-XL
        logger.debug("big")
        
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big"))
        wandb.log({'ppl': ppl, 'total_ppl': total_ppl})

    if 'toxicity' in metricset:
        logger.debug("toxicity")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score(generations_df, perspective_file=output_dir / (output_file+".toxicity"))
        wandb.log({'avg_max_toxicity': avg_max_toxicity, 'toxic_probability_p': toxic_probability_p,
                'avg_toxicity': avg_toxicity, 'toxic_probability_s': toxic_probability_s})
            
    if 'toxicity-energy' in metricset:
        logger.debug("toxicity-energy")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_energy(generations_df, toxicity_file=output_dir / (output_file+".toxicity_energy"))
        wandb.log({'avg_max_toxicity_energy': avg_max_toxicity, 'toxic_probability_p_energy': toxic_probability_p,
                'avg_toxicity_energy': avg_toxicity, 'toxic_probability_s_energy': toxic_probability_s})
        # with open(output_dir / output_file, 'a') as fo:
        #     fo.write(f'[energy model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
        #     logger.debug(f'[energy model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
            
    if 'toxicity-mucola' in metricset:
        logger.debug("toxicity-mucola")
        (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_mucola(generations_df, toxicity_file=output_dir / (output_file+".toxicity_mucola"))
        wandb.log({'avg_max_toxicity_mucola': avg_max_toxicity, 'toxic_probability_p_mucola': toxic_probability_p,
                'avg_toxicity_mucola': avg_toxicity, 'toxic_probability_s_mucola': toxic_probability_s})
        # with open(output_dir / output_file, 'a') as fo:
        #     fo.write(f'[mucola model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')
        #     logger.debug(f'[mucola model] avg_max_toxicity = {avg_max_toxicity}, toxicity prob prompt = {toxic_probability_p}, avg_toxicity = {avg_toxicity}, toxicity prob={toxic_probability_s}\n')

    if "dist-n" in metricset:
        logger.debug("dist-n")
        dist1, dist2, dist3 = distinctness(generations_df)
        wandb.log({'dist-1': dist1, 'dist-2': dist2, 'dist-3': dist3})
        # # write output results
        # with open(output_dir / output_file, 'a') as fo:
        #     for i, dist_n in enumerate([dist1, dist2, dist3]):
        #         fo.write(f'dist-{i+1} = {dist_n}\n')
        #         print(f'dist-{i+1} = {dist_n}')
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_path', help='Run path of a previous run to eval.')
    parser.add_argument('--outfile', help='Path to the outputs file for eval.')
    parser.add_argument('--metrics', default='toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n', help='comma-separated string of a list of metrics for eval.')
    
    args = parser.parse_args()
    
    main(args.run_path, args.outfile, args.metrics)