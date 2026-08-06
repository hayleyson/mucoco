import argparse
import gc
import logging
import os
import json
import dotenv
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb
from laser_edit.evaluation.metrics import (
    conditional_perplexity,
    distinctness,
    fluency_classify,
    formality_score_ext,
    formality_score_int,
    repetition,
    sentiment_classify_big,
    sentiment_classify_own2,
    toxicity_score,
    # toxicity_score_energy,
    toxicity_score_int,
    # toxicity_score_mucola,
    nli_score,
    sentiment_classify_gpt4o,
    contents_preservation_metrics,
    save_qualitative_results,
    set_consistency_score,
    set_consistency_score_gpt,
    save_qualitative_results,
    toxicity_nli_joint_rates_from_saved_eval,
)

## logging-related
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("le")
logger.setLevel(logging.DEBUG)

dotenv.load_dotenv()

def rename_df_for_nli(dataframe, col_name='premise'):
    # rename target column for evaluation
    result_df = dataframe.copy()
    result_df['prompt'] = dataframe['prompt'].apply(lambda x: {'text': x[col_name]})
    return result_df[['prompt', 'generations']]


def _normalize_sc_gpt_dataset(task: str) -> str:
    """Map eval --task names to parse_set_text dataset ids (lconvqa | set_nli)."""
    if task in {'lconvqa', 'set_lconvqa', 'set-lconvqa', 'vqa', 'convqa'}:
        return 'lconvqa'
    if task in {'set_nli', 'set-nli', 'set_snli', 'set-snli'}:
        return 'set_nli'
    return task


def load_sc_energy_llm_edit_outputs(generations_file_path, dataset='lconvqa'):
    with open(generations_file_path, 'r') as f:
        # Assuming the JSONL has one large dict with 'pred' key containing all examples
        data = json.loads(f.readline())
    preds = data['pred']
    
    formatted_texts = []
    
    if dataset in ['lconvqa', 'set_lconvqa', 'vqa']:
        for gen_list in tqdm(preds, desc='Loading LLM edit results', mininterval=5):
            text_parts = []
            for pair in gen_list:
                if len(pair) >= 2:
                    q = pair[0]
                    a = pair[1]
                    if q is None:
                        continue
                    if a is not None:
                        # Reconstruct the original string format used in lconvqa text generations
                        formatted = f"{q} The answer is {a}."
                    else:
                        # E.g. "**Input Text:**\n" and other anomalies produced by the model
                        formatted = str(q) 
                    text_parts.append(formatted.strip())
            
            full_text = " ".join(text_parts).strip()
            formatted_texts.append({
                "prompt": {"text": ""},
                "generations": [{"text": full_text}]
            })
            
        return pd.DataFrame(formatted_texts)
    elif dataset in ['set_nli']:
        for gen_list in tqdm(preds, desc='Loading LLM edit results', mininterval=5):
            text_parts = []
            for text in gen_list:
                text_parts.append(text.strip())
            
            full_text = " ".join(text_parts).strip()
            formatted_texts.append({
                "prompt": {"text": ""},
                "generations": [{"text": full_text}]
            })
        print("Loaded dataset preview:")
        print(pd.DataFrame(formatted_texts).head())
        return pd.DataFrame(formatted_texts)
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def release_cuda_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_generation_evaluation(run_path, generations_file_path, metrics, **kwargs):
    """
    kwargs: 
    - includes "formality_model_path", "formality_model_type" for formality-int score
    - includes "sentiment_model_path", "sentiment_model_type" for sentiment-int score
    - includes "toxicity_model_path", "toxicity_model_type" for toxicity-int score
    - includes "source_file_path" for contents-preservation score
    """
    
    if kwargs.get('set_consistency_llm_edit_output', False):
        generations_df = load_sc_energy_llm_edit_outputs(generations_file_path, dataset=kwargs.get('task', 'lconvqa'))
    else:
        generations_df = pd.read_json(generations_file_path, lines=True) 
    if type(generations_df['prompt'].values[0]) == str:
        generations_df['prompt'] = generations_df['prompt'].apply(lambda x: {'text': x})
    if type(generations_df['generations'].values[0][0]) == str:
        generations_df['generations'] = generations_df['generations'].apply(lambda x: [ {'text': y} for y in x])
    logger.debug(generations_df.shape)


    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(metricset)
    
    if run_path != "": ## if wandb run path is provided.
        api = wandb.Api()
        run = api.run(run_path)
    
    output_file = f"{generations_file_path.split('/')[-1]}-results.txt"
    task = kwargs.get("task", "")
    target_style = kwargs.get("target_style", "")
        
    output_dir = Path(os.path.dirname(generations_file_path))
    if os.path.exists(output_dir / output_file):
        fp = open(output_dir / output_file, 'a')
        fp.write('-'*50+'\n')
    else:
        fp = open(output_dir / output_file, 'w')

    if "ppl-qwen" in metricset: #GPT2-XL
        logger.debug("big")
        eval_model_name = "Qwen/Qwen2.5-14B"
        torch.cuda.empty_cache()
        eval_model = None
        eval_tokenizer = None
        generations_df2 = None
        try:
            eval_model = AutoModelForCausalLM.from_pretrained(eval_model_name, dtype = torch.float16, device_map="auto")
            eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
            torch.cuda.empty_cache()
            if task=='nli':
                # generations_df2 = rename_df_for_nli(generations_df, 'premise')
                generations_df2 = generations_df.copy()
                generations_df2['prompt'] = [{"text":''}] * len(generations_df2)
            else:
                generations_df2 = generations_df.copy()
            with torch.no_grad():
                ppl, total_ppl = conditional_perplexity(generations_df2, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big-qwen"))
            if run_path != "":
                run.summary.update({'ppl_qwen': ppl, 'total_ppl_qwen': total_ppl})
            fp.write(f'ppl_qwen: {ppl}, total_ppl_qwen: {total_ppl}\n')
        finally:
            release_cuda_memory(eval_model, eval_tokenizer, generations_df2)
        

    if "ppl-big" in metricset: #GPT2-XL
        logger.debug("big")
        torch.cuda.empty_cache()
        eval_model = None
        eval_tokenizer = None
        generations_df2 = None
        try:
            eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
            eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
            torch.cuda.empty_cache()
            if task=='nli':
                # generations_df2 = rename_df_for_nli(generations_df, 'premise')
                generations_df2 = generations_df.copy()
                generations_df2['prompt'] = [{"text":''}] * len(generations_df2)
            else:
                generations_df2 = generations_df.copy()
            with torch.no_grad():
                ppl, total_ppl = conditional_perplexity(generations_df2, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big"))
            if run_path != "":
                run.summary.update({'ppl': ppl, 'total_ppl': total_ppl})
            fp.write(f'ppl: {ppl}, total_ppl: {total_ppl}\n')
        finally:
            release_cuda_memory(eval_model, eval_tokenizer, generations_df2)
    
    if 'nli' in metricset:
        logger.debug("nli-ensemble")
        # generations_df2 = rename_df_for_nli(generations_df, 'premise')
        generations_df2 = generations_df.copy()
        print(generations_df2.head())
        (avg_nli_entail, avg_nli_neutral, avg_nli_contradiction, contradiction_proba, entail_proba, neutral_proba) = nli_score(generations_df2, task=task, write_file=output_dir / (output_file+".nli"), device='cuda')
        if run_path != "":
            run.summary.update({'avg_nli_entail': avg_nli_entail, 'avg_nli_neutral': avg_nli_neutral,
                'avg_nli_contradiction': avg_nli_contradiction, 'contradiction_proba': contradiction_proba, 'entail_proba': entail_proba, 'neutral_proba': neutral_proba})
        fp.write(f'avg_nli_entail: {avg_nli_entail}, avg_nli_neutral: {avg_nli_neutral}, avg_nli_contradiction: {avg_nli_contradiction}, contradiction_proba: {contradiction_proba}, entail_proba: {entail_proba}, neutral_proba: {neutral_proba}\n')

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
        (avg_max_toxicity_int, toxic_probability_p_int, avg_toxicity_int, toxic_probability_s_int) = toxicity_score_int(generations_df, output_dir / (output_file+".toxicity_int"), device,
                                                                                                        kwargs['toxicity_model_path'], kwargs['toxicity_model_type'])
        if run_path != "":
            run.summary.update({'avg_max_toxicity_int': avg_max_toxicity_int, 'toxic_probability_p_int': toxic_probability_p_int,
                'avg_toxicity_int': avg_toxicity_int, 'toxic_probability_s_int': toxic_probability_s_int})
        fp.write(f'avg_max_toxicity_int: {avg_max_toxicity_int}, toxic_probability_p_int: {toxic_probability_p_int}, avg_toxicity_int: {avg_toxicity_int}, toxic_probability_s_int: {toxic_probability_s_int}\n')

    if 'formality-ext' in metricset:
        logger.debug("formality-external")
        avg_formality, formal_proba = formality_score_ext(generations_df, output_dir / (output_file+".formality_ext"), device)
        if run_path != "":
            run.summary.update({'avg_formality': avg_formality, 'formal_proba': formal_proba})
        fp.write(f'avg_formality: {avg_formality}, formal_proba: {formal_proba}\n')
        
    if 'formality-int' in metricset:
        logger.debug("formality-internal")
        avg_formality_int, formal_proba_int = formality_score_int(generations_df, output_dir / (output_file+".formality_int"), device, 
                                                          kwargs['formality_model_path'], kwargs['formality_model_type'])
        if run_path != "":
            run.summary.update({'avg_formality_int': avg_formality_int, 'formal_proba_int': formal_proba_int})
        fp.write(f'avg_formality_int: {avg_formality_int}, formal_proba_int: {formal_proba_int}\n')
        
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
       
    if 'sentiment-gpt4o' in metricset:
        logger.debug("sentiment-gpt4o")
        positive_proba_gpt4o, std_positive_proba_p_gpt4o = sentiment_classify_gpt4o(generations_df, output_dir / (output_file+".sentiment_gpt4o"))
        if run_path != "":
            # run.summary.update({"avg_sentiment": None, "positive_proba": None, 
            #                 "avg_positive_proba_p": None, "std_positive_proba_p": None, 
            #                 "positive_proba_p_avg": None, "positive_proba_s": None})
            run.summary.update({'positive_proba_gpt4o': positive_proba_gpt4o, 
                                'positive_proba_std_gpt4o': std_positive_proba_p_gpt4o})
        fp.write(f'positive_proba_gpt4o: {positive_proba_gpt4o}, positive_proba_std_gpt4o: {std_positive_proba_p_gpt4o}\n')
       
        
    if 'sentiment-int' in metricset:
        logger.debug("sentiment-internal")
        positive_proba_int, std_positive_proba_p_int, avg_positivity_int = sentiment_classify_own2(generations_df, output_dir / (output_file+".sentiment_int"),
                                                          kwargs['sentiment_model_path'], kwargs['sentiment_model_type'])
        if run_path != "":
            # run.summary.update({"avg_sentiment_int": None, "positive_proba_int": None, 
            #                     "avg_positive_proba_p_int": None, "std_positive_proba_p_int": None, 
            #                     "positive_proba_p_avg_int": None, "positive_proba_s_int": None})
            run.summary.update({'positive_proba_int': positive_proba_int, 
                                'positive_proba_p_std_int': std_positive_proba_p_int,
                                'avg_positivity_int': avg_positivity_int})
        fp.write(f'positive_proba_int: {positive_proba_int}, positive_proba_p_std_int: {std_positive_proba_p_int}, avg_positivity_int: {avg_positivity_int}\n')

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
        generations_df2 = generations_df.copy()
        fluency = fluency_classify(generations_df2, output_dir / (output_file+".fluency"))
        if run_path != "":
            run.summary.update({'fluent_proba': fluency})
        fp.write(f'fluent_proba: {fluency}\n')
        
    if "set-consistency" in metricset:
        logger.debug("set-consistency")
        
        device = 'cuda'
        
            
        if task in ['nli', 'set_nli', 'set-nli', 'set_snli', 'set-snli']:
            config_path = 'laser_edit/set_consistency_energy/params_set_nli.yaml'
        elif task in ['vqa', 'lconvqa', 'convqa', 'set-lconvqa', 'set_lconvqa']:
            config_path = 'laser_edit/set_consistency_energy/params_set_lconvqa.yaml'
        
        avg_sc_score, contradiction_rate = set_consistency_score(generations_df, output_dir / (output_file+".sc"), device, config_path)
        if run_path != "":
            run.summary.update({'avg_sc_score': avg_sc_score, 'consistent_proba': 1 - contradiction_rate})
        fp.write(f'avg_sc_score: {avg_sc_score}, consistent_proba: {1 - contradiction_rate}\n')

    if "set-consistency-clsf" in metricset:
        logger.debug("set-consistency-clsf")
        
        device = 'cuda'
        
            
        if task in ['nli', 'set_nli', 'set-nli', 'set_snli', 'set-snli']:
            config_path = 'laser_edit/set_consistency_energy/params_set_nli_clsf.yaml'
        elif task in ['vqa', 'lconvqa', 'convqa', 'set-lconvqa', 'set_lconvqa']:
            config_path = 'laser_edit/set_consistency_energy/params_set_lconvqa_clsf.yaml'
        
        avg_sc_score, contradiction_rate = set_consistency_score(generations_df, output_dir / (output_file+".sc_clsf"), device, config_path)
        if run_path != "":
            run.summary.update({'avg_sc_score_clsf': avg_sc_score, 'consistent_proba_clsf': 1 - contradiction_rate})
        fp.write(f'avg_sc_score_clsf: {avg_sc_score}, consistent_proba_clsf: {1 - contradiction_rate}\n')
    
    if "set-consistency-gpt" in metricset:
        logger.debug("set-consistency-gpt")
        
        device = 'cuda'
        sc_gpt_dataset = _normalize_sc_gpt_dataset(kwargs['task'])
        
        cons_prop = 1 - set_consistency_score_gpt(generations_df, "gpt-5-mini", output_dir / (output_file+".sc_gpt"), dataset=sc_gpt_dataset)
        if run_path != "":
            run.summary.update({'consistent_proba_gpt': cons_prop})
        fp.write(f'consistent_proba_gpt: {cons_prop}\n')

        if run_path != "":
            run.summary.update({'consistent_proba_gpt': cons_prop})
        
    if "set-consistency-gpt-5.4-mini" in metricset:
        logger.debug("set-consistency-gpt-5.4-mini")
        
        device = 'cuda'
        sc_gpt_dataset = _normalize_sc_gpt_dataset(kwargs['task'])
        
        cons_prop = 1 - set_consistency_score_gpt(generations_df, "gpt-5.4-mini", output_dir / (output_file+".sc_gpt_5.4_mini"), dataset=sc_gpt_dataset)
        if run_path != "":
            run.summary.update({'consistent_proba_gpt_5.4_mini': cons_prop})
        fp.write(f'consistent_proba_gpt_5.4_mini: {cons_prop}\n')

        if run_path != "":
            run.summary.update({'consistent_proba_gpt_5.4_mini': cons_prop})
        
    if "set-consistency-gpt-5.4" in metricset:
        logger.debug("set-consistency-gpt-5.4")
        
        device = 'cuda'
        sc_gpt_dataset = _normalize_sc_gpt_dataset(kwargs['task'])
        
        cons_prop = 1 - set_consistency_score_gpt(generations_df, "gpt-5.4", output_dir / (output_file+".sc_gpt_5.4"), dataset=sc_gpt_dataset)
        if run_path != "":
            run.summary.update({'consistent_proba_gpt_5.4': cons_prop})
        fp.write(f'consistent_proba_gpt_5.4: {cons_prop}\n')

        if run_path != "":
            run.summary.update({'consistent_proba_gpt_5.4': cons_prop})
        
        
    if "contents-preservation" in metricset:
        logger.debug("contents-preservation")
        
        torch.cuda.empty_cache()
        # if (task == "formality") and (target_style == 'informal'):
        #     kwargs['source_file_path'] = '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal'
        # elif (task == "formality") and (target_style == 'formal'):
        #     kwargs['source_file_path'] = '/home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal'
        # print(kwargs['source_file_path'])
        sbleu_score, sbert_score, sbert_preserved_prop, sbert_preserved_count = contents_preservation_metrics(kwargs['source_file_path'],
                                                                    generations_df, 
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
    
    if "h1" in metricset:
        logger.debug("h1")
        ## metric for sweep
        ## harmonic mean of fluency and constraint satisfaction rate
        if task == 'toxicity':
            constraint_sat = 1 - toxic_probability_s
        elif (task == 'sentiment') and (target_style == 'positive'):
            constraint_sat = positive_proba
        elif (task == 'sentiment') and (target_style == 'negative'):
            constraint_sat = 1 - positive_proba
        elif (task == 'formality') and (target_style == 'formal'):
            constraint_sat = formal_proba
        elif (task == 'formality') and (target_style == 'informal'):
            constraint_sat = 1 - formal_proba
        elif task == 'nli':
            constraint_sat = 1 - contradiction_proba
        elif task in ['set_nli', 'set-nli', 'set_snli', 'set-snli', 'lconvqa', 'set_lconvqa', 'sc_energy', 'set-lconvqa']:
            constraint_sat = 1 - contradiction_rate
        logger.info(f"task: {task}, target_style: {target_style}, constraint_sat: {constraint_sat}, fluency: {fluency}")
        
        h1 = (2 * fluency * constraint_sat) / (fluency + constraint_sat)
        
        if run_path != "":
            run.summary.update(
                        {"h1": h1}
                    )
        fp.write(f"h1: {h1}\n")
          
    
    if "qual" in metricset:
        if task == 'toxicity':
            constraint_suffix = 'toxicity'
        elif task == 'sentiment':
            constraint_suffix = 'sentiment_ext'
        elif task == 'formality':
            constraint_suffix = 'formality_ext'
        elif task == 'nli':
            constraint_suffix = 'nli'
        elif task in ['set_nli', 'set-nli', 'set_snli', 'set-snli', 'lconvqa', 'set_lconvqa', 'sc_energy', 'set-lconvqa']:
            constraint_suffix = 'sc_gpt'
            
        save_qualitative_results(task,
                                kwargs['source_file_path'], 
                                generations_df, 
                                str(output_dir / (output_file+".ppl-big-qwen")) if os.path.exists(output_dir / (output_file+".ppl-big-qwen")) else str(output_dir / (output_file+".ppl-big")), 
                                str(output_dir / (output_file+f".{constraint_suffix}")), 
                                str(output_dir / (output_file+".sbertscore")),
                                str(output_dir / (output_file+".xlsx")))
    
    if "nli_toxicity_joint" in metricset:
        logger.debug("nli_toxicity_joint")
        nli_toxicity_joint_rates = toxicity_nli_joint_rates_from_saved_eval(generations_file_path)
        if run_path != "":
            run.summary.update({'nli_toxicity_joint_pct_both': nli_toxicity_joint_rates['pct_both'],
                                'nli_toxicity_joint_pct_only_nontoxic': nli_toxicity_joint_rates['pct_only_nontoxic'],
                                'nli_toxicity_joint_pct_only_consistent': nli_toxicity_joint_rates['pct_only_consistent'],
                                'nli_toxicity_joint_pct_neither': nli_toxicity_joint_rates['pct_neither']})
        fp.write(f"nli_toxicity_joint_pct_both: {nli_toxicity_joint_rates['pct_both']}, nli_toxicity_joint_pct_only_nontoxic: {nli_toxicity_joint_rates['pct_only_nontoxic']}, nli_toxicity_joint_pct_only_consistent: {nli_toxicity_joint_rates['pct_only_consistent']}, nli_toxicity_joint_pct_neither: {nli_toxicity_joint_rates['pct_neither']}\n")

            
    if run_path != "":
        run.update()
    fp.close()      
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_path', default='', help='Wandb run path of the decoding run to eval.')
    parser.add_argument('--generations_file_path', help='Path to the decoding outputs file for eval.')
    parser.add_argument('--metrics', default='toxicity,toxicity-energy,toxicity-mucola,ppl-big,dist-n', help='comma-separated string of a list of metrics for eval.')
    parser.add_argument('--sentiment_model_path', type=str, help='path to sentiment energy model (or binary classifier) used for decoding')
    parser.add_argument('--sentiment_model_type', type=str, choices=['RobertaCustomForSequenceClassification', 'AutoModelForSequenceClassification'], help='the type of sentiment energy model (or binary classifier) used for decoding')
    parser.add_argument('--formality_model_path', type=str, help='path to formality energy model (or binary classifier) used for decoding')
    parser.add_argument('--formality_model_type', type=str, choices=['RobertaCustomForSequenceClassification', 'AutoModelForSequenceClassification'], help='the type of formality energy model (or binary classifier) used for decoding')
    parser.add_argument('--toxicity_model_path', type=str, help='path to toxicity energy model (or binary classifier) used for decoding')
    parser.add_argument('--toxicity_model_type', type=str, choices=['RobertaCustomForSequenceClassification', 'AutoModelForSequenceClassification'], help='the type of toxicity energy model (or binary classifier) used for decoding')
    parser.add_argument('--source_file_path', type=str, help='path to the original generations file')
    parser.add_argument('--task', type=str, default='', help='Task name (e.g. set_nli, set_lconvqa); used for metric routing and SC-energy loader.')
    parser.add_argument('--target_style', type=str, default='', help='Target style for constraint-side metrics when applicable.')
    parser.add_argument('--set_consistency_llm_edit_output', action='store_true',
                        help='Interpret generations_file_path as SC-energy LLM-edit JSON (pred list format).')

    
    args = parser.parse_args()
    
    
    run_generation_evaluation(
        args.run_path,
        args.generations_file_path,
        args.metrics,
        sentiment_model_path=args.sentiment_model_path,
        sentiment_model_type=args.sentiment_model_type,
        formality_model_path=args.formality_model_path,
        formality_model_type=args.formality_model_type,
        toxicity_model_path=args.toxicity_model_path,
        toxicity_model_type=args.toxicity_model_type,
        source_file_path=args.source_file_path,
        task=args.task,
        target_style=args.target_style,
        set_consistency_llm_edit_output=args.set_consistency_llm_edit_output,
    )