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
    
    # evaluate_main("hayleyson/sentiment-decoding/8f1lniw9",
    #     "/data/hyeryung/mucoco/outputs/sentiment/negative_gpt2/8f1lniw9/outputs_epsilon0.999994.txt",
    #     metrics="contents-preservation",
    #     task = "sentiment",
    #     target_style="negative",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl",)
    
    # evaluate_main("hayleyson/sentiment-decoding/kvuoipf5",
    #     "/data/hyeryung/mucoco/outputs/sentiment/negative_gpt2/kvuoipf5/outputs_epsilon0.999994.txt",
    #     metrics="contents-preservation",
    #     task = "sentiment",
    #     target_style="negative",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl",)
    
    # evaluate_main("hayleyson/sentiment-decoding/jce4qaqa",
    #     "/data/hyeryung/mucoco/outputs/sentiment/positive_gpt2/jce4qaqa/outputs_epsilon0.9999994.txt",
    #     metrics="contents-preservation",
    #     task = "sentiment",
    #     target_style="positive",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_positive_threshold_778.jsonl",)
    
    # evaluate_main("hayleyson/sentiment-decoding/usi7f6qx",
    #     "/data/hyeryung/mucoco/outputs/sentiment/positive_gpt2/usi7f6qx/outputs_epsilon0.9999994.txt",
    #     metrics="contents-preservation",
    #     task = "sentiment",
    #     target_style="positive",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_positive_threshold_778.jsonl",)
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/new_module/data/logical-consistency/remove_em/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl",
    #     metrics="fluency",
    #     task = "nli",
    #     target_style="consistent",)
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/nli/s2ztkwrh/outputs_epsilon0.99.jsonl",
    #     metrics="fluency",
    #     task = "nli",
    #     target_style="consistent",)

    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/nli/pkides1c/outputs_epsilon0.99.jsonl",
    #     metrics="fluency",
    #     task = "nli",
    #     target_style="consistent",)
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/sentiment/mucola/below_negative_threshold_827/bdos3o04/outputs_epsilon-5.22.txt",
    #     metrics="ppl-qwen,contents-preservation",
    #     task = "sentiment",
    #     target_style="negative",)
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/sentiment/mucola/below_negative_threshold_827/zgxgqzim/outputs_epsilon-2.txt",
    #     metrics="ppl-qwen,contents-preservation",
    #     task = "sentiment",
    #     target_style="negative",)
    
    # evaluate_main("hayleyson/sentiment_gbi/bdos3o04",
    #     "/data/hyeryung/mucoco/outputs/sentiment/mucola/below_negative_threshold_827/bdos3o04/outputs_epsilon-5.22.txt",
    #     metrics="ppl-qwen,contents-preservation",
    #     task = "sentiment",
    #     target_style="negative",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl",)   
    
    # evaluate_main("hayleyson/sentiment_gbi/zgxgqzim",
    #     "/data/hyeryung/mucoco/outputs/sentiment/mucola/below_negative_threshold_827/zgxgqzim/outputs_epsilon-2.txt",
    #     metrics="ppl-qwen,contents-preservation",
    #     task = "sentiment",
    #     target_style="negative",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_negative_threshold_827.jsonl",)    

    # evaluate_main("hayleyson/formality_gbi/rdeb7d6n",
    #     "/data/hyeryung/mucoco/outputs/formality/mucola/informal2formal/rdeb7d6n/outputs_epsilon-2.txt",
    #     metrics="ppl-qwen",
    #     task = "formality",
    #     target_style="formal",
    #     source_file_path="/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal",)    

    # evaluate_main("hayleyson/formality_gbi/l9nui59p",
    #     "/data/hyeryung/mucoco/outputs/formality/mucola/informal2formal/l9nui59p/outputs_epsilon-0.17.txt",
    #     metrics="ppl-qwen",
    #     task = "formality",
    #     target_style="formal",
    #     source_file_path="/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal",)  
    
    # evaluate_main("",
    #     "/data/hyeryung/mixmatch/output_samples/form_em_clsf_0_roberta/disc_frm_new_data_form_em_test_sh8_len_b_sc_r_inf_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_300.0_date_08_06_2024_20_36_40/opt_samples.jsonl",
    #     metrics="ppl-qwen",
    #     task = "formality",
    #     target_style="formal",
    #     source_file_path="/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal",)  
    
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/sentiment/bolt/clsf/pos.dev_set_below_positive_threshold.jsonl",
    #     metrics="sentiment-ext,ppl-qwen,fluency,dist-n,repetition,contents-preservation",
    #     task = "sentiment",
    #     target_style="positive",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_positive_threshold_778_unraveled.jsonl",)  
      
    # evaluate_main("",
    #     "/data/hyeryung/mucoco/outputs/sentiment/mixmatch/clsf_pos_merged/pos.dev_set_below_positive_threshold.jsonl",
    #     metrics="sentiment-ext,ppl-qwen,fluency,dist-n,repetition,contents-preservation",
    #     task = "sentiment",
    #     target_style="positive",
    #     source_file_path="/data/hyeryung/mucoco/new_module/data/sentiment/dev_set_below_positive_threshold_778_unraveled.jsonl")
    
    # for n_iter in range(0, 10):
    #     evaluate_main("",
    #         f"/data/hyeryung/mucoco/outputs/formality/formal/7v5u2lr9/outputs_epsilon0.74.txt.{n_iter}",
    #         metrics="ppl-qwen",
    #         task = "formality",
    #         target_style="formal",
    #         source_file_path="/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal",)  
        
    # run_ids = [
    #         "zoxb26aa",
    #         "dt87s3dq",
    #         "i5djrjop",
    #         "yutq1ewg",
    #         "w0k8c7bo",
    #         "9qtowiq1",
    #         "etn81kci",
    #         "r4i9vv5t",
    #         "0g1gdqq9"
    #         ]
    # for run_id in run_ids:
    #     evaluate_main(f"hayleyson/formality-decoding/{run_id}",
    #         f"/data/hyeryung/mucoco/outputs/formality/formal/{run_id}/outputs_epsilon0.74.txt",
    #         metrics="ppl-qwen",
    #         task = "formality",
    #         target_style="formal",
    #         source_file_path="/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal",)
   
    # fpaths = ['outputs/nli/s2ztkwrh/outputs_epsilon0.99.txt',
    #           'outputs/nli/pkides1c/outputs_epsilon0.99.txt',
    #           ] 
    # for fpath in fpaths:
    #     evaluate_main(f"hayleyson/nli-decoding/{fpath.split('/')[-2]}",
    #         fpath,
    #         metrics="contents-preservation",
    #         task = "nli",
    #         target_style="consistent",
    #         source_file_path="new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl",)

   
    # fpaths = ['outputs/saehee/qwen/nli_em_removed/6_nli_edited_38454.jsonl_total_0', 
    #           'outputs/saehee/qwen/nli_em_removed/9_nli_loc_edit_38546.jsonl'] 
    # for fpath in fpaths:
    #     evaluate_main("",
    #         fpath,
    #         metrics="nli",
    #         task = "nli",
    #         target_style="consistent",
    #         source_file_path="new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl",)
    
    # fpaths = ['outputs/saehee/qwen/2_tox_edited_38592.jsonl_total_0', 
    #           'outputs/saehee/qwen/5_tox_loc_edit_38576.jsonl'] 
    # for fpath in fpaths:
    #     evaluate_main("",
    #         fpath,
    #         metrics="toxicity,ppl-qwen,fluency,dist-n,repetition,contents_preservation",
    #         task = "toxicity",
    #         target_style="nontoxic",
    #         source_file_path="/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl",)

    # evaluate_main("/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_0_9.jsonl",
    #     fpath,
    #     metrics="nli,ppl-qwen,fluency,dist-n,repetition",
    #     task = "nli",
    #     target_style="consistent",
    #     source_file_path="",)

    # evaluate_main("",
    #     "/data/hyeryung/mucoco/new_module/data/logical-consistency/anli-r2-test_prompt_4.jsonl",
    #     metrics="nli,fluency",
    #     task = "nli",
    #     target_style="consistent",
    #     source_file_path="",)
    
    evaluate_main("hayleyson/sentiment_gbi/j4w5dhzq",
        "/data/hyeryung/mucoco/outputs/sentiment/mucola/below_positive_threshold_778/j4w5dhzq/outputs_epsilon-2.txt",
        metrics="ppl-qwen",
        task = "sentiment",
        target_style="positive",
        source_file_path="",)

    evaluate_main("hayleyson/sentiment_gbi/ww4jv4ou",
        "/data/hyeryung/mucoco/outputs/sentiment/mucola/below_positive_threshold_778/ww4jv4ou/outputs_epsilon-5.22.txt",
        metrics="ppl-qwen",
        task = "sentiment",
        target_style="positive",
        source_file_path="",)
