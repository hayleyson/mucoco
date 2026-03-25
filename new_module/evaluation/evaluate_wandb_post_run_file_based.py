import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_wandb import evaluate_main


# toxicity task
# toxicity,ppl-qwen,fluency,repetition,dist-n,contents-preservation
# /data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl

# nli task
# nli,ppl-qwen,fluency,repetition,dist-n,contents-preservation
# new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl

# sentiment task
# sentiment-ext,ppl-qwen,fluency,repetition,dist-n,contents-preservation

if __name__ == "__main__":
        
        # evaluate_main("hayleyson/toxicity-decoding/j18pi8ab",
        #     "/data/hyeryung/mucoco/outputs/toxicity/llm/j18pi8ab/outputs_epsilon0.95.txt.0",
        #     metrics="contents-preservation",
        #     task="toxicity",
        #     target_style="nontoxic",
        #     source_file_path="/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl")
        
        # evaluate_main("",
        #     "/data/hyeryung/mucoco/outputs/toxicity/mixmatch/mask_disc_max_len_12_jigsaw_clsf_data_detoxic_em_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_15.0_gamma_0.0_theta_100.0_date_14_12_2024_03_07_10/opt_samples.jsonl",
        #     metrics="contents-preservation",
        #     task="toxicity",
        #     target_style="nontoxic",
        #     source_file_path="/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl")
        
        # evaluate_main("",
        #     "/data/hyeryung/mucoco/outputs/toxicity/mucola/gpt35/fcm49jjy/outputs_epsilon-2.26.txt",
        #     metrics="contents-preservation",
        #     task="toxicity",
        #     target_style="nontoxic",
        #     source_file_path="/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl")
        
        # evaluate_main("",
        #     "/data/hyeryung/mucoco/outputs/llmedit/results_saehee_2024/qwen/2_tox_edited_38592.jsonl_total_0",
        #     metrics="contents-preservation",
        #     task="toxicity",
        #     target_style="nontoxic",
        #     source_file_path="/data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl")
    
    # evaluate_main(
    #         "",
    #         "outputs/sc_energy/set_lconvqa/debug/outputs.txt",
    #         "set-consistency,ppl-qwen,fluency,dist-n,repetition,contents-preservation",
    #         task="set_lconvqa",
    #         source_file_path="/home/hyeryung/data/mucoco/new_module/data/convqa/processed_data/lconvqa_test1_edited_only.jsonl"
    #     )  
#     evaluate_main(
#             "",
#             "new_module/data/set_nli/eval2/set_nli_eval2.jsonl",
#             "set-consistency,ppl-qwen,fluency,dist-n,repetition,contents-preservation",
#             task="set_snli",
#             source_file_path="/home/hyeryung/data/mucoco/new_module/data/set_nli/processed_data/set_nli_test1_edited_only.jsonl"
#     )  
    evaluate_main(
            "",
            "outputs/toxicity/scope/toxicity/disc1/scope_gen_nontoxic_scope_nontoxic_gpt2xl_1e-5_epoch20_disc1.jsonl",
            "toxicity,ppl-qwen,fluency,repetition,dist-n,contents-preservation",
            task="toxicity",
            source_file_path="/home/hyeryung/data/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl"
    ) 
#     evaluate_main(
#             "",
#             "outputs/sc_energy/set_snli/ebm/btah19b5/outputs.txt",
#             "set-consistency-clsf",
#             task="set_snli",
#     ) 
#     evaluate_main(
#             "",
#             "outputs/sc_energy/set_snli/classifier/b48qkoxt/outputs.txt",
#             "set-consistency-clsf",
#             task="set_snli",
#     ) 
#     evaluate_main(
#             "",
#             "new_module/data/set_nli/processed_data/set_nli_test1_edited_only.jsonl",
#             "set-consistency-clsf",
#             task="set_snli",
#     ) 
        
