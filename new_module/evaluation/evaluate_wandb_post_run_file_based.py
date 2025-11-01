import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_wandb import evaluate_main


if __name__ == "__main__":
    
    
    # evaluate_main("",
    #               "/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_noprompt_150.jsonl",
    #               metrics="ppl-qwen",
    #               task="toxicity",
    #               target_style="nontoxic")
    
    # evaluate_main("",
    #               "/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl",
    #               metrics="toxicity,ppl-qwen,fluency,repetition,dist-n",
    #               task="toxicity",
    #               target_style="nontoxic")
    
    # evaluate_main("",
    #             "/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl",
    #             metrics="ppl-qwen,fluency,repetition,dist-n",
    #             task="toxicity",
    #             target_style="nontoxic")
    
#     evaluate_main("",
#             "/data/hyeryung/mixmatch/output_samples/detoxic_r/mask_disc_max_len_12_jigsaw_clsf_data_detoxic_em_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_100.0_date_08_06_2024_17_01_54/opt_samples.jsonl",
#             metrics="ppl-qwen",
#             task="toxicity",
#             target_style="nontoxic")
    
#     evaluate_main("",
#             "/data/hyeryung/BOLT/detoxic/detoxic/gen_len20.jsonl",
#             metrics="ppl-qwen",
#             task="toxicity",
#             target_style="nontoxic")

    # evaluate_main(
    #         "",
    #         "new_module/data/set_nli/processed_data/set_nli_test.jsonl",
    #         "set-consistency,dist-n,repetition,fluency",
    #         task="set_nli",
    #     )  
    
    evaluate_main(
            "",
            "new_module/data/convqa/processed_data/lconvqa_test.jsonl",
            "set-consistency,dist-n,repetition,fluency",
            task="lconvqa",
        )  
    
    
