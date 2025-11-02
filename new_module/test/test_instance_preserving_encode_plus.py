"""
Goal: test if instance_preserving_encode_plus works correctly
"""
# import
import time
import json
import math
import re
import yaml
import os
from argparse import Namespace

import transformers
import huggingface_hub
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader

from new_module.em_training.nli.models import EncoderModel  
from new_module.locate.new_locate_utils import LocateMachine, LocateMachine4SCE
from new_module.set_consistency_energy.energynets.energynet import energynet
import new_module.losses as lossbuilder


def load_sc_energy_model(config_path, folder_path, model_path, time_key, task, device, locate_option):
        
    model_config = yaml.load(open(config_path), 
                                Loader=yaml.FullLoader)
    dataset = 'set_nli' if task == 'nli' else 'lconvqa'
    
    model_config['dataset'] = dataset
    model_config['task'] = task
    model_config['folder_path'] = folder_path
    model_config['model_path'] = model_path
    model_config['time_key'] = time_key
    
    if locate_option == "attention":
        model_config['locate']['type'] = "attention"
    elif locate_option == "grad_norm":
        model_config['locate']['type'] = "gradnorm"

    energy_net = energynet(params=model_config)
    model_object = torch.load(model_config["model_path"], 
                                map_location=device,
                                weights_only=True)
    energy_net.load_state_dict(model_object['state_dict'], strict=False)
    if 'threshold' in model_object:
        energy_net.threshold = model_object['threshold']
    
    energy_net.eval()
    energy_net.to(device)
    
    return energy_net, model_config

   
def test_instance_preserving_encode_plus(test_case):
    
    # 환경 설정
    task = "nli"
    max_num_tokens = 7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    locate_option = "attention"
    
    # 모델과 토크나이저 로드
    if task == "nli": # NOTE. different from actual nli task. It is set_nli.
        
        config_path = 'new_module/set_consistency_energy/params.yaml'
        folder_path = 'new_module/set_consistency_energy/results/nli/set_nli/46853'
        model_path = os.path.join(folder_path, 'SetCon-roberta-no-triplet-False-fg_tot.pth')
        time_key = '46853'
        
        model, model_config = load_sc_energy_model(config_path, folder_path, model_path, time_key, "nli", device, locate_option)

    elif task == "vqa":
        
        config_path = 'new_module/set_consistency_energy/params.yaml'
        folder_path = 'new_module/set_consistency_energy/results/vqa/lconvqa/1225068'
        model_path = os.path.join(folder_path, 'SetCon-roberta-no-triplet-False-fg_tot.pth')
        time_key = '1225068'

        model, model_config = load_sc_energy_model(config_path, folder_path, model_path, time_key, "vqa", device, locate_option)

    print("start locate")

    # LocateMachine 초기화
    locator = LocateMachine4SCE(model_config, model, task)

    infile = [{"prompt": {"text": ""}, 
            "generations": [{"text": test_case}]}]

    for line_idx, line in enumerate(infile):
        # JSON 형식으로 변환
        data = line
        prompt = data['prompt']['text']
        generations = data['generations']
        
        masked_generations = []
        # generations 내의 각 text에 대해 LocateMachine 적용
        for gen_idx, generation in enumerate(generations):
            
            text = generation['text']
            # locate_main 적용
            print(f"text: {text}")
            masked_text = locator.locate_main([text],  
                                                max_num_tokens=max_num_tokens, 
                                                unit='word')
            # masked 결과를 generation에 추가 (기존 key나 새로운 key 사용 가능)
            print(f"masked_text: {masked_text}")
            generation['text'] = masked_text[0]  # locate_main은 리스트를 반환하므로 첫 번째 값 선택
            masked_generations.append(generation)
    
        print(f"masked_generations: {masked_generations}")
    
    
    
    
if __name__ == "__main__":
    
    
    test_case = """<s>Edited Text: Either a man holds up a sign that reads "tattoo" as many people walk near him or a man is holding a sign and yelling "tattoo". No man is holding a sign and yelling "tattoo"."""
    
    test_instance_preserving_encode_plus(test_case)