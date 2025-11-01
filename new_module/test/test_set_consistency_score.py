import os

import torch
import pandas as pd

from evaluation.prompted_sampling.evaluate import set_consistency_score

def test_set_consistency_score():
    torch.cuda.empty_cache()
    generations_df = pd.read_json('outputs/sc_energy/nli/debug/outputs.txt', lines=True)
    output_file = 'outputs/sc_energy/nli/debug/outputs-results.txt.sc'
    device = 'cuda'
    config_path = 'new_module/set_consistency_energy/params.yaml'
    folder_path = 'new_module/set_consistency_energy/results/nli/set_nli/46853'
    model_path = os.path.join(folder_path, 'SetCon-roberta-no-triplet-False-fg_tot.pth')
    time_key = '46853'
    task = 'nli'
    avg_sc_score, cons_prop = set_consistency_score(generations_df, output_file, device, config_path, folder_path, model_path, time_key, task)
    print(f"Average set consistency score: {avg_sc_score}")
    print(f"Set consistency proportion: {cons_prop}")
    
if __name__ == "__main__":
    test_set_consistency_score()