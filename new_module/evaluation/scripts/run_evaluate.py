import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_pipeline import run_generation_evaluation


# toxicity task
# toxicity,ppl-qwen,fluency,repetition,dist-n,contents-preservation
# /data/hyeryung/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl

# nli task
# nli,ppl-qwen,fluency,repetition,dist-n,contents-preservation
# new_module/data/logical-consistency/anli-r2-test_prompt_4_below_consistent_threshold_3105.jsonl

# sentiment task
# sentiment-ext,ppl-qwen,fluency,repetition,dist-n,contents-preservation

# set consistency task
# set-consistency,set-consistency-gpt,ppl-qwen,fluency,dist-n,repetition,contents-preservation

if __name__ == "__main__":
        
    # run_generation_evaluation(
    #     "",
    #     "outputs/sc_energy/set_lconvqa/ebm/bs48a98o/outputs.txt",
    #     "set-consistency-gpt",
    #     source_file_path="new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
    #     task="set_lconvqa",
    # ) 
    
    run_generation_evaluation(
        "",
        "outputs/sc_energy/set_lconvqa/ebm/zg7pzbsw/outputs.txt",
        "set-consistency-gpt",
        source_file_path="new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
        task="set_lconvqa",
    ) 

    # run_generation_evaluation(
    #     "",
    #     "outputs/sc_energy/set_lconvqa/6vzkroqi/outputs_qwen2.5_7B_s0_p0.96_refined.txt",
    #     "set-consistency,set-consistency-clsf,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
    #     source_file_path="new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
    #     task="set_lconvqa",
    # ) 

    # run_generation_evaluation("",
    #     "outputs/sc_energy/set_lconvqa/6vzkroqi/outputs_qwen2.5_7B_s5_p0.1_refined.txt",
    #     "set-consistency,set-consistency-clsf,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
    #     source_file_path="new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
    #     task="set_lconvqa",
    # )  

    # run_generation_evaluation(
    #     "",
    #     "outputs/sc_energy/set_lconvqa/6vzkroqi/outputs_qwen2.5_7B_s5_p0.96_refined.txt",
    #     "set-consistency,set-consistency-clsf,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
    #     source_file_path="new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
    #     task="set_lconvqa",
    # )  