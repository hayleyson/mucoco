import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_wandb import evaluate_main


if __name__ == "__main__":


    # evaluate_main("",
    #               "/data/hyeryung/mixmatch/output_samples/detoxic/mask_disc_max_len_12_jigsaw_clsf_data_detoxic_em_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_100.0_date_24_04_2024_15_45_58/opt_samples.txt",
    #               'contents-preservation',#formality-ext,ppl-big,dist-n,repetition,fluency,qual
    #               source_file_path='/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal',
    #               task = "formality",
    #               target_style="formal")
    
    # evaluate_main("",
    #               "/data/hyeryung/mixmatch/output_samples/detoxic/mask_disc_max_len_12_jigsaw_clsf_data_detoxic_em_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_100.0_date_24_04_2024_15_45_58/opt_samples.jsonl",
    #               'toxicity,ppl-big,dist-n,repetition,fluency',
    #               source_file_path='/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl',
    #               task = "toxicity",
    #               target_style="nontoxic")
    
    # evaluate_main("",
    #               "/data/hyeryung/mixmatch/output_samples/detoxic/mask_disc_max_len_12_jigsaw_em_data_detoxic_em_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_100.0_date_03_05_2024_14_16_06/opt_samples.jsonl",
    #               'toxicity,ppl-big,dist-n,repetition,fluency',
    #               source_file_path='/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl',
    #               task = "toxicity",
    #               target_style="nontoxic")
    
    # evaluate_main("",
    #               "/data/hyeryung/mixmatch/output_samples/detoxic/mask_disc_max_len_12_jigsaw_em_data_detoxic_em_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_100.0_date_03_05_2024_14_16_06/opt_samples.jsonl",
    #               'toxicity,ppl-big,dist-n,repetition,fluency',
    #               source_file_path='/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl',
    #               task = "toxicity",
    #               target_style="nontoxic")
    
    # evaluate_main("",
    #               "/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_noprompt_150.jsonl",
    #               'toxicity,ppl-big,fluency,dist-n,repetition',
    #               source_file_path='/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl',
    #               task = "toxicity",
    #               target_style="nontoxic")
    
    # evaluate_main("",
    #               "/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl",
    #               'fluency,dist-n,repetition',
    #               source_file_path='/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl',
    #               task = "toxicity",
    #               target_style="nontoxic")
    
    # evaluate_main("hayleyson/toxicity-decoding/q7tlrfcl",
    #             "/data/hyeryung/mucoco/outputs/toxicity/llm/q7tlrfcl/outputs_epsilon0.9.txt",
    #             'toxicity',
    #             task = "toxicity",
    #             target_style="nontoxic")
    
    evaluate_main("hayleyson/toxicity-decoding/r7kykwge",
            "/data/hyeryung/mucoco/outputs/toxicity/llm/r7kykwge/outputs_epsilon0.9.txt",
            'toxicity',
            task = "toxicity",
            target_style="nontoxic")