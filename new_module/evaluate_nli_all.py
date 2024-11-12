import os
import sys
from glob import glob

import pandas as pd
import time
import wandb

# setting for import
os.chdir('/data3/saeheeeom/set_consistency/mucoco')
sys.path.append(os.path.abspath('.'))

from new_module.evaluation.evaluate_wandb import evaluate_main



if __name__ == "__main__":
    #filepath = '/data3/saeheeeom/set_consistency/mucoco/new_module/llm_experiments/edit_with_llm/edit/old(1009)/total_results/'
    filepath = '/data3/saeheeeom/set_consistency/mucoco/new_module/llm_experiments/edit_with_llm/edit/old(1009)/total_results/preprocessed/'

    filenames = [
#"zero_gemma_27711_both2.jsonl",         "zero_phi_27784_masked.jsonl",
#"zero_mistral_27810_both2.jsonl",      "zero_phi_27784_notmasked.jsonl",
#"zero_gemma_27711_masked.jsonl",       
#"zero_qwen_27661_both2.jsonl",
#"zero_gemma_27711_notmasked.jsonl",  "zero_mistral_27810_masked.jsonl",
#"zero_llama_27721_masked.jsonl",     "zero_mistral_27810_notmasked.jsonl",  
#"zero_qwen_27661_masked.jsonl",
#"zero_llama_27721_notmasked.jsonl",  "zero_phi_27784_both2.jsonl",    "zero_qwen_27661_notmasked.jsonl",
#"zero_llama_27757_both2.jsonl"


#"zero_gemma_27711_both2_preprocess.jsonl","zero_gemma_27711_masked_preprocess.jsonl", "zero_gemma_27711_notmasked_preprocess.jsonl",
#"zero_qwen_27661_both2_preprocess.jsonl","zero_qwen_27661_masked_preprocess.jsonl","zero_qwen_27661_notmasked_preprocess.jsonl",
#"zero_phi_27784_masked_preprocess.jsonl",
"zero_mistral_27810_both2_preprocess.jsonl",      "zero_phi_27784_notmasked_preprocess.jsonl",
"zero_mistral_27810_masked_preprocess.jsonl",
"zero_llama_27721_masked_preprocess.jsonl",     "zero_mistral_27810_notmasked_preprocess.jsonl",  
"zero_llama_27721_notmasked_preprocess.jsonl",  "zero_phi_27784_both2_preprocess.jsonl",          
"zero_llama_27757_both2_preprocess.jsonl"
]
    #"""
    for filename in filenames:
        #if 'notmasked' in filename:
            #continue
        start_time = time.time()
        filename = filepath + filename
        if 'gemma' in filename:
            sourcefile = "zero_gemma_27711_notmasked_preprocess.jsonl"
        elif 'llama' in filename:
            sourcefile = "zero_llama_27721_notmasked_preprocess.jsonl"
        elif 'mistral' in filename:
            sourcefile = "zero_mistral_27810_notmasked_preprocess.jsonl"
        elif 'phi' in filename:
            sourcefile = "zero_phi_27784_notmasked_preprocess.jsonl"
        elif 'qwen' in filename:
            sourcefile = "zero_qwen_27661_notmasked_preprocess.jsonl"
        sourcefile = filepath + sourcefile
        
        print("========================================")
        print("evaluating", filename)
        evaluate_main("",
                filename,
                #metrics="toxicity,ppl-big,dist-n,repetition,fluency,contents-preservation",
                #metrics="fluency",
                metrics="ppl-big",
                #metrics='contents-preservation',
                task = "nli",
                #target_style="nontoxic",
                source_file_path=sourcefile)
        end_time = time.time()
        print(f"Time taken: {(end_time - start_time)/60} minutes")

    # evaluate_main("hayleyson/toxicity-decoding/r7kykwge",
    #         "/data/hyeryung/mucoco/outputs/toxicity/llm/r7kykwge/outputs_epsilon0.9.txt",
    #         metrics="ppl-big,fluency,dist-n,repetition",
    #         task = "toxicity",
    #         target_style="nontoxic")