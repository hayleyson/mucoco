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


def contents_preservation_metrics(sources_file,outputs_file,results_file,task):
    
    if task in ['toxicity','sentiment']:
        sources = pd.read_json(sources_file, lines=True)
        sources.prompt=sources.prompt.apply(lambda x: x['text'])
        
        predictions = pd.read_json(outputs_file, lines=True)
        predictions.prompt=predictions.prompt.apply(lambda x: x['text'])
        if task=='toxicity':
            source_predictions=pd.merge(sources,predictions,on='prompt',how='inner',suffixes=('_source','_prediction'))
        elif task=='sentiment':
            source_predictions=pd.concat([sources,predictions],axis=1)
            source_predictions=source_predictions.iloc[:, [0,1,4]].copy()
            source_predictions.columns=['prompt','generations_source','generations_prediction']
            
        prompt_list=[]
        source_list=[]
        prediction_list=[]
        for _, row in source_predictions.iterrows():
            prompt_list.extend([row.prompt]*len(row.generations_source))
            for i in range(len(row.generations_source)):
                source_list.append(row.generations_source[i]['text'])
                prediction_list.append(row.generations_prediction[i]['text'])
        source_predictions_=pd.DataFrame({'prompt':prompt_list,'source':source_list,'prediction':prediction_list})
        
    elif task=='formality':
        with open(sources_file,'r') as f:
            sources = [line.rstrip('\n') for line in f.readlines()]
            
        predictions = pd.read_json(outputs_file, lines=True)
        predictions = predictions.explode('generations')
        predictions['generations']=predictions['generations'].apply(lambda x: x['text'])
        
        source_predictions_ = pd.DataFrame({'source': sources, 'prediction': predictions['generations'].tolist()}) 
        
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

    return sbleu_score, sbert_score*100, sbert_preserved_prop, sbert_preserved_count


def unravel(outputs_df):
    outputs_df=outputs_df.explode('generations',ignore_index=True)
    outputs_df['prompt']=outputs_df['prompt'].apply(lambda x: x['text'])
    outputs_df['generations']=outputs_df['generations'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
    outputs_df = outputs_df.dropna().reset_index(drop=True)
    return outputs_df

def unravel_toxicity_data(df):
    df['toxicity']=df['allresponses'].apply(lambda x: [x[0]['attributeScores']['TOXICITY']['summaryScore']['value'] for x in list(x.values())])
    df=df.explode('toxicity',ignore_index=True)
    return df

def save_qualitative_results(task,
                             source_file_path, 
                             outputs_file_path, 
                             ppl_results_path, 
                             constraint_results_path, 
                             contents_prsrv_results_path,
                             qual_results_path):
    
    
    # read files
    if (task=='toxicity') or (task=='sentiment'):
        source = pd.read_json(source_file_path, lines=True)
    elif (task=='formality'):
        with open(source_file_path, 'r') as f:
            source = [_line.rstrip('\n') for _line in f.readlines()]
    
    outputs = pd.read_json(outputs_file_path, lines=True)
    ppl = pd.read_csv(ppl_results_path, header=None)
    if (task=='toxicity') or (task=='sentiment'):
        constraint_sat = pd.read_json(constraint_results_path, lines=True)
    elif (task=='formality'):
        constraint_sat = pd.read_csv(constraint_results_path, header=None)
    contents_prsrv=pd.read_csv(contents_prsrv_results_path)


    # preprocess files
    ## key (row index), prompt, gen 
    if (task=='toxicity'): 
        source = unravel(source)
    elif (task=='sentiment'):
        source = unravel(source)
        source = source[['prompt','generations']].copy()
    elif (task=='formality'):
        source = pd.DataFrame({'prompt': ["" for _ in range(len(source))], 'generations': source})
    outputs = unravel(outputs)

    ## key (row index), value
    ppl = ppl.iloc[:, 0].copy()

    if task == 'toxicity':
        constraint_sat = unravel_toxicity_data(constraint_sat)
        constraint_sat = constraint_sat[['toxicity']].copy()
        constraint_sat['toxicity'] = 1-constraint_sat['toxicity']
    elif task == 'sentiment': 
        constraint_sat.loc[constraint_sat['label']=='NEGATIVE', 'score'] = constraint_sat.loc[constraint_sat['label']=='NEGATIVE', 'score'].apply(lambda x: 1-x)
        constraint_sat = constraint_sat['score'].copy()
    elif task == 'formality':
        constraint_sat = constraint_sat.iloc[:, 0].copy()
                                
    contents_prsrv = contents_prsrv['sbert_score'].copy()

    final_df=pd.concat([source,outputs[['generations']],ppl,constraint_sat,contents_prsrv],axis=1,ignore_index=True)

    final_df.columns=['prompt','original','edited','ppl','constraint_sat','sbert_score']
    final_df.to_excel(qual_results_path,index=False)


def evaluate_main(run_path, generations_file_path, metrics, **kwargs):
    """
    kwargs: 
    - includes "formality_model_path", "formality_model_type" for formality-int score
    - includes "sentiment_model_path", "sentiment_model_type" for sentiment-int score
    - includes "toxicity_model_path", "toxicity_model_type" for toxicity-int score
    - includes "source_file_path" for contents-preservation score
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
        
        # if run.state != 'finished':
        #     try:
        #         if task == 'toxicity':
        #             assert len(generations_df) == 250
        #         elif task == 'formality':
        #             assert len(generations_df) == 1416
        #         elif task == 'sentiment':
        #             assert len(generations_df) == 15
        #     except:
        #         raise Exception(f"The number of generations is not correct. {len(generations_df)} while task is {task}")
        #     ## if the run state is not finished but the number of generations are complete -> finish the run
        #     run1 = wandb.init(project=run_path.split('/')[1], id=run_path.split('/')[-1], resume="must")
        #     run1.finish()
        #     del run1
        ## update model_tag if it is not set
        model_tag = run.config.get('model_tag', None)
        if (model_tag is None) or (model_tag == ''):
            run.config['model_tag'] = 'em' if ('energy-training' in model_path) else 'clsf'
            if (task == 'formality') and ('gyafc' in model_path):
                run.config['model_tag'] += '-gyafc'
        
        target_style = run.config.get("target_style", "")
    else:
        output_file = "results.txt"
        task = kwargs.get("task", "")
        target_style = kwargs.get("target_style", "")
        
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
        
    if "contents-preservation" in metricset:
        logger.debug("contents-preservation")
        
        torch.cuda.empty_cache()
        if (task == "formality") and (target_style == 'informal'):
            kwargs['source_file_path'] = '/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal'
        elif (task == "formality") and (target_style == 'formal'):
            kwargs['source_file_path'] = '/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal'
        print(kwargs['source_file_path'])
        sbleu_score, sbert_score, sbert_preserved_prop, sbert_preserved_count = contents_preservation_metrics(kwargs['source_file_path'],
                                                                    generations_file_path, 
                                                                    str(output_dir / output_file),
                                                                    task)
        
        if run_path != "":
            run.summary.update(
                {
                    "sbleu": sbleu_score,
                    "sbert": sbert_score,
                    "sbert_preserved_prop": sbert_preserved_prop,
                    "sbert_preserved_count": sbert_preserved_count
                }
            )
        fp.write(f"sbleu: {sbleu_score}\n")
        fp.write(f"sbert_score: {sbert_score}, sbert_preserved_prop: {sbert_preserved_prop}, sbert_preserved_count: {sbert_preserved_count}\n")
            
    if run_path != "":
        run.update()
    fp.close()        
    
    if "qual" in metricset:
        if task == 'toxicity':
            constraint_suffix = 'toxicity'
        elif task == 'sentiment':
            constraint_suffix = 'sentiment_ext'
        elif task == 'formality':
            constraint_suffix = 'formality_ext'
            
        save_qualitative_results(task,
                                kwargs['source_file_path'], 
                                generations_file_path, 
                                str(output_dir / (output_file+".ppl-big")), 
                                str(output_dir / (output_file+f".{constraint_suffix}")), 
                                str(output_dir / (output_file+".sbertscore")),
                                str(output_dir / (output_file+".xlsx")))
    
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
    parser.add_argument('--source_file_path', type=str, help='path to the original generations file')

    
    args = parser.parse_args()
    
    
    evaluate_main(args.generations_file_path, args.metrics, args.run_path, 
             sentiment_model_path=args.sentiment_model_path, sentiment_model_type=args.sentiment_model_type,
             formality_model_path=args.formality_model_path, formality_model_type=args.formality_model_type,
             toxicity_model_path=args.toxicity_model_path, toxicity_model_type=args.toxicity_model_type)