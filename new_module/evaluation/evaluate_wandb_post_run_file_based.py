import os
import sys
from glob import glob

import pandas as pd

import wandb
from new_module.evaluation.evaluate_wandb import evaluate_main


if __name__ == "__main__":
    
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
#     evaluate_main(
#             "",
#             "new_module/data/set_nli/eval2/set_nli_eval2.jsonl",
#             "set-consistency,ppl-qwen,fluency,dist-n,repetition",
#             task="set_snli",
#             source_file_path="/home/hyeryung/data/mucoco/new_module/data/set_nli/processed_data/set_nli_test1_edited_only.jsonl"
#     ) 
    evaluate_main(
            "",
            "outputs/sc_energy/set_snli/ebm/btah19b5/outputs.txt",
            "set-consistency-clsf",
            task="set_snli",
    ) 
    evaluate_main(
            "",
            "outputs/sc_energy/set_snli/classifier/b48qkoxt/outputs.txt",
            "set-consistency-clsf",
            task="set_snli",
    ) 
    evaluate_main(
            "",
            "new_module/data/set_nli/processed_data/set_nli_test1_edited_only.jsonl",
            "set-consistency-clsf",
            task="set_snli",
    ) 
        
