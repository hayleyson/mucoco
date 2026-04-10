## evaluation
import re
import time
from datetime import datetime
from glob import glob
import joblib
import pandas as pd
import random 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langdetect import detect
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
    nli_score,
    sentiment_classify_gpt4o,
    contents_preservation_metrics,
    save_qualitative_results
)

def is_english(text):
    try:
        return detect(text) == 'en'  # 'en'은 영어를 의미
    except:
        return False  # 예외 발생 시 False 처리
    


with open('/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332_index.txt', 'r') as f:
    indices = f.read().split()
indices = [int(x) for x in indices]
located_data = pd.read_json('/home/hyeryung/data/mucoco/new_module/locate/locate_num_tokens_eda/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_nontoxic_locate_max_7.jsonl', lines=True)
located_data = located_data.explode('generations').reset_index(drop=True)
located_data = located_data.loc[located_data['generations'].apply(len) != 0].reset_index(drop=True)

located_data['prompt'] = located_data['prompt'].apply(lambda x: x['text'])
located_data['masked_sentences'] = located_data['generations'].apply(lambda x: x['text'])
located_data = located_data.loc[indices,:].copy()
print(f"Number of total samples: {len(located_data)}")
all_source_texts = located_data['prompt'].tolist()
all_masked_sentences = located_data['masked_sentences'].tolist()

num_test_samples = 50
random.seed(999)
idxes_for_test = random.sample(range(len(all_source_texts)),num_test_samples)
prompts = [all_source_texts[i] for i in idxes_for_test]

result_list = sorted(glob('new_module/decoding_result_using_v*.pkl'))
time_list = sorted(glob('new_module/decoding_time_using_v*.txt'))

outf = open(f"new_module/decoding_result_all_{datetime.today().strftime('%Y%m%d%H%M%S')}.csv", 'w')
outf.write('method,,fluency_em,time,time_per_sample,ppl(qwen),total_ppl(qwen),fluency,avg_toxicity,toxic_proba\n')

for result_path, time_path in zip(result_list, time_list):
    
    print('------------------------------------')
    method = re.findall('v[0-9]+', result_path)[0]
    fluency_em = result_path.split('/')[-1].split('_')[-1][:-4]
    print(method, fluency_em)
    
    with open(time_path, 'r') as f:
        execution_time = float(f.read().strip())
    
    # load generation file
    results = joblib.load(result_path)
    results = sum(results, [])

    generations_df = pd.DataFrame({
        'prompt': [{'text': x} for x in prompts],
        'generations': [[{'text': x}] for x in results]
    })

    device = 'cuda'
    torch.cuda.empty_cache()
    eval_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-14B', torch_dtype = torch.float16).to('cuda')
    eval_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B')
    torch.cuda.empty_cache()

    with torch.no_grad():
        ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=None)
    print(ppl, total_ppl)
    del eval_model, eval_tokenizer

    fluency = fluency_classify(generations_df, None)
    print(fluency)

    toxicity_model_path = '/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint/'
    toxicity_model_type = 'AutoModelForSequenceClassification'
    (avg_max_toxicity, toxic_probability_p, avg_toxicity, toxic_probability_s) = toxicity_score_int(generations_df, None, device,
                                                                                                toxicity_model_path, toxicity_model_type)
    print(avg_toxicity, toxic_probability_s)

    outf.write(f"{method},,{fluency_em},{execution_time},{execution_time/num_test_samples},{ppl},{total_ppl},{fluency},{avg_toxicity},{toxic_probability_s}\n")


outf.close()