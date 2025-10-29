###########################################################
# Package import 
import joblib
import argparse
import os
import yaml
import random
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

import new_module.losses as lossbuilder
from new_module.locate.new_locate_utils import LocateMachine4SCE
from new_module.set_consistency_energy.energynets.energynet import energynet
from new_module.new_decode_utils import analyze_span_lengths_and_count, editing_4sce, editing_with_delete_variable_replace

random.seed(42)

# Set global variables
root_dir = 'new_module/set_consistency_energy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###########################################################


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
args = parser.parse_args()

task = args.task

###########################################################

# Load models => Define loss functions
# 1) MLM
mlm = AutoModelForMaskedLM.from_pretrained('roberta-base')
mlm.eval()
mlm.to(device)
mlm_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# 2) Causal LM
# causal_lm = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
causal_lm = AutoModelForCausalLM.from_pretrained('gpt2-large')
causal_lm.eval()
causal_lm.half()
causal_lm.to(device)
# causal_lm_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
causal_lm_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
causal_lm_tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})

# 3) Energy Net

model_config = yaml.load(open('new_module/set_consistency_energy/params.yaml'), 
                               Loader=yaml.FullLoader)
if task == 'nli':
    
    model_config['dataset'] = 'set_nli'
    model_config['task'] = 'nli'
    model_config['folder_path'] = 'new_module/set_consistency_energy/results/nli/set_nli/46853'
    model_config['model_path'] = os.path.join(model_config['folder_path'], 'SetCon-roberta-no-triplet-False-fg_tot.pth')
    model_config['time_key'] = '46853'

elif task == 'vqa':    
    pass # params already set for vqa task


energy_net = energynet(params=model_config)
energy_net.load_state_dict(torch.load(model_config["model_path"], 
                                      map_location=model_config['device'],
                                      weights_only=True)['state_dict'], strict=False)
if 'threshold' in torch.load(model_config["model_path"],
                             map_location=model_config['device'],
                             weights_only=True):
    energy_net.threshold = torch.load(model_config["model_path"],
                             map_location=model_config['device'],
                             weights_only=True)['threshold']
energy_net.eval()
energy_net.to(device)

energy_net_tokenizer = energy_net.representation_model.tokenizer
energy_net_tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})


# Wrap models into loss functions

@dataclass
class LossArgs:
    length_normalize: bool = True
    alpha: float = 1.0
    AR_temperature: float = 1.0
    AR_top_k: int = 0
    AR_top_p: float = 0.96
    max_output_length: int = 20
    task: str = None
    device: str = None

loss_args = LossArgs(task=task,
                     device=device)

lossfns = []
losses = ['gpt2', 'sc_energy']
models = [causal_lm, energy_net]
tokenizers = [causal_lm_tokenizer, energy_net_tokenizer]

for i, loss in enumerate(losses):
    lossfns.append(
        lossbuilder.build_loss(
            loss,
            models[i],
            tokenizers[i],
            loss_args,
        )
    )


###########################################################

# Set up LocateMachine4SCE

locator = LocateMachine4SCE(model_config, energy_net, task)

###########################################################

# Set up sample data

# Load dataset
if (task == 'nli') or (task == 'set_nli'):
    data_path = 'new_module/data/set_nli/processed_data/set_nli_test_for_locate_edit.pickle'
elif (task == 'vqa') or (task == 'convqa') or (task == 'lconvqa') or (task == 'set_lconvqa'):
    data_path = 'new_module/data/convqa/processed_data/lconvqa_test_for_locate_edit.pickle'
else:
    raise ValueError(f"Task {task} not supported")

data = joblib.load(data_path)

# Filter only inconsistent data
incon = list(filter(lambda x: x[-1] == 'incon', data))

# Filter data with length greater than 9 sentences 
# incon = list(filter(lambda x: len(x[2]) > 9, incon))

# Select one sample
source_text = ""

if (source_text == "") and (lossfns[0].tokenizer.bos_token is not None):
    source_text = lossfns[0].tokenizer.bos_token
elif (source_text == "") and (lossfns[0].tokenizer.bos_token is None):
    source_text = " "



# original_text = [random.sample(incon, 1)[0][0]]
original_text = [incon[0][0]]
running_text = original_text

print(f"original_text: {original_text}")


###########################################################
# Locate

masked_text = locator.locate_main(running_text, max_num_tokens=7, unit='word')
print(f"masked_text: {masked_text}")

###########################################################
# Edit


config = {'task': task, 
          'device': device,
          'losses': ['gpt2', 'sc_energy'],
          'loss_weights': [1, 1], 
          'target_label_ids': [1, 1],
          'selection_criteria': 'allsat_primary', 
          'beam_size': 5,
          'min_epsilons': [0.95],
          'k_per_location': 5,
          'max_tokens_per_span': 3, 
          'consider_prompt_for_cand_gen': True,
          }

_, span_lengths = analyze_span_lengths_and_count(masked_text[0])
    
if len(span_lengths) > 0:
    
    final_hypotheses_curr, new_best_weighted_loss_curr, new_best_allsat_curr, new_best_logging_loss_curr = \
                editing_4sce(source_text, 
                     running_text[0], 
                     masked_text[0], 
                     span_lengths,
                     mlm, 
                     mlm_tokenizer, 
                     lossfns, 
                     config, 
                     batch_size=32, 
                     post_context_mode="original")
                
        # editing_with_delete_variable_replace(source_text, 
        #                                      masked_text[0], 
        #                                      span_lengths, 
        #                                      mlm, 
        #                                      mlm_tokenizer, 
        #                                      lossfns, 
        #                                      config, 
        #                                      batch_size=32)
        
        
    print(f"final_hypotheses_curr: {final_hypotheses_curr}")