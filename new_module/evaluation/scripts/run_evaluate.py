import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_pipeline import run_generation_evaluation


# toxicity task
# toxicity,ppl-qwen,fluency,repetition,dist-n,contents-preservation
# /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl

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
    #     "/home/hyeryung/data/mucoco/outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen2.5-7b-instruct_w_ebm_locate_edit_result.jsonl",
    #     "ppl-qwen,dist-n,contents-preservation",
    #     source_file_path="/home/hyeryung/data/mucoco/new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
    #     task="set_lconvqa",
    #     set_consistency_llm_edit_output=True,
    # )
    
    # run_generation_evaluation(
    #     "",
    #     "/home/hyeryung/data/mucoco/outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen2.5-7b-instruct_w_gt_locate_edit_result.jsonl",
    #     "ppl-qwen,dist-n,contents-preservation",
    #     source_file_path="/home/hyeryung/data/mucoco/new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
    #     task="set_lconvqa",
    #     set_consistency_llm_edit_output=True,
    # )
    
    # run_generation_evaluation(
    #     "",
    #     "/home/hyeryung/data/mucoco/outputs/sc_energy/set_lconvqa/llm/testset_incon_300/set_lconvqa_qwen2.5-7b-instruct_w_self_locate_edit_result.jsonl",
    #     "ppl-qwen,dist-n,contents-preservation",
    #     source_file_path="/home/hyeryung/data/mucoco/new_module/data/convqa/locate/testset_incon_300/lconvqa_testset_incon_300.jsonl",
    #     task="set_lconvqa",
    #     set_consistency_llm_edit_output=True,
    # )
    
    
    run_generation_evaluation(
        "",
        "/home/hyeryung/data/mucoco/outputs/sc_energy/set_nli/llm/testset_incon_300/set_nli_qwen2.5-7b-instruct_w_ebm_locate_edit_result.jsonl",
        "set-consistency,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
        source_file_path="new_module/data/set_nli/testset_incon_300/set_nli_testset_incon_300.jsonl",
        task="set_nli",
        set_consistency_llm_edit_output=True,
    )
    
    
    # run_generation_evaluation(
    #     "",
    #     "/home/hyeryung/data/mucoco/outputs/sc_energy/set_nli/ebm/wv1tlwb0/outputs.txt",
    #     "set-consistency,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
    #     source_file_path="new_module/data/set_nli/testset_incon_300/set_nli_testset_incon_300.jsonl",
    #     task="set_nli",
    #     # set_consistency_llm_edit_output=True,
    # )