import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_wandb import evaluate_main


if __name__ == "__main__":
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/multi/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_neg_150.jsonl",
    #     # metrics="toxicity,sentiment-ext,sentiment-gpt4o,formality-ext,ppl-qwen,fluency,dist-n,repetition",
    #     metrics="sentiment-gpt4o,fluency,dist-n,repetition",
    #     task = "toxicity",
    #     target_style="nontoxic")
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/toxicity/llm/q59dutqt/outputs_epsilon0.9.txt",
    #     metrics="sentiment-ext,sentiment-gpt4o,formality-ext,ppl-qwen",
    #     task = "toxicity",
    #     target_style="nontoxic")
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/toxicity/llm/tymxk7th/outputs_epsilon0.9.txt",
    #     metrics="sentiment-ext,sentiment-gpt4o,formality-ext,ppl-qwen",
    #     task = "toxicity",
    #     target_style="nontoxic")
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/toxicity/llm/bx3p1fwj/outputs_epsilon0.9.txt",
    #     metrics="sentiment-ext,sentiment-gpt4o,formality-ext,ppl-qwen",
    #     task = "toxicity",
    #     target_style="nontoxic")
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/toxicity/llm/zgfsopqw/outputs_epsilon0.9.txt",
    #     metrics="sentiment-ext,sentiment-gpt4o,formality-ext,ppl-qwen",
    #     task = "toxicity",
    #     target_style="nontoxic")
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/new_module/llm_experiments/edit_with_llm/baselm_gens/gemma2-2b-it/edited_gemma_2b_0shot_37112_cleaned.jsonl",
    #     metrics="toxicity,ppl-qwen,fluency,dist-n,repetition,contents_preservation",
    #     source_file_path='/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl',
    #     task = "toxicity",
    #     target_style="nontoxic")
    
    evaluate_main("",
        "/data/hyeryung/mucoco/new_module/llm_experiments/edit_with_llm/baselm_gens/gemma2-9b-it/edited_gemma_9b_0shot_37111_cleaned.jsonl",
        metrics="toxicity,ppl-qwen,fluency,dist-n,repetition,contents_preservation",
        source_file_path='/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl',
        task = "toxicity",
        target_style="nontoxic")