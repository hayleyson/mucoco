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
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/new_module/data/sentiment/dev_set.jsonl",
    #     metrics="sentiment-int,sentiment-ext,sentiment-gpt4o,ppl-qwen,fluency,dist-n,repetition",
    #     task = "sentiment",
    #     target_style="positive",
    #     sentiment_model_path="/data/hyeryung/loc_edit/models/roberta-base-yelp-sentiment-classifier-energy-training/step_81900_best_checkpoint",
    #     sentiment_model_type="AutoModelForSequenceClassification",)
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/new_module/data/formality/informal.jsonl",
    #     metrics="ppl-qwen",
    #     task = "formality",
    #     target_style="formal",)
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4.jsonl",
    #     metrics="nli,ppl-qwen,fluency,dist-n,repetition",
    #     task = "nli",
    #     target_style="consistent",)
    
    evaluate_main("hayleyson/sentiment-decoding/8f1lniw9",
        "/data/hyeryung/mucoco/outputs/sentiment/negative_gpt2/8f1lniw9/outputs_epsilon0.999994.txt",
        metrics="contents-preservation",
        task = "sentiment",
        target_style="negative",
        source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl",)
    
    evaluate_main("hayleyson/sentiment-decoding/kvuoipf5",
        "/data/hyeryung/mucoco/outputs/sentiment/negative_gpt2/kvuoipf5/outputs_epsilon0.999994.txt",
        metrics="contents-preservation",
        task = "sentiment",
        target_style="negative",
        source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl",)
    
    evaluate_main("hayleyson/sentiment-decoding/jce4qaqa",
        "/data/hyeryung/mucoco/outputs/sentiment/positive_gpt2/jce4qaqa/outputs_epsilon0.9999994.txt",
        metrics="contents-preservation",
        task = "sentiment",
        target_style="positive",
        source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_positive_threshold_778.jsonl",)
    
    evaluate_main("hayleyson/sentiment-decoding/usi7f6qx",
        "/data/hyeryung/mucoco/outputs/sentiment/positive_gpt2/usi7f6qx/outputs_epsilon0.9999994.txt",
        metrics="contents-preservation",
        task = "sentiment",
        target_style="positive",
        source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_positive_threshold_778.jsonl",)