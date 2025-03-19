# copy of llm_generate
# toxicity jsonl 파일 처리 위함

import joblib
import json
import argparse
import os
import time
import re
import sys

import torch

import pandas as pd
import numpy as np

import time
import random

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



import huggingface_hub
huggingface_token = os.getenv("HF_TOKEN")
huggingface_hub.login(token=huggingface_token)




pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-7B-Instruct",
    device="cuda",  # replace with "mps" to run on a Mac device
    do_sample=True,
)


from tqdm import tqdm


def generate_answer(text, max_length):
  outputs = pipe(text, max_new_tokens=max_length, num_return_sequences=30)
  response = [o["generated_text"] for o in outputs]
  # get only the output
  new_response = []
  for res in response:
    res = res[len(text):]
    new_response.append(res)
  response = new_response
  return response


device= "cuda" if torch.cuda.is_available() else "cpu"


print("Loading data...")

premises = []

snli_test_path = "/home/saeheeeom/data/set_consistency/data/pplm_prompts.jsonl"
print("evaluation data:", snli_test_path)
print("====================================================================")

with open(snli_test_path,'r') as f:
    raw_data = f.readlines()
    for line in raw_data:
        tmp_list = []
        line = json.loads(line)

        p = line['prompt']['text']
        premises.append(p)
f.close()



# start generation

all_gens = []
for i in [50, 100, 150]:
    print("=========== start generation! =============")
    max_length = i
    start_time = time.time()
    objs = []
    for j, premise in enumerate(tqdm(premises)):
      prompt = f"Complete the sentence in a negative tone: {premise}"
      generations = generate_answer(prompt, max_length)
      json_obj = {'prompt':{'text': premise}, 'generations': [{'text': g} for g in generations]}
      objs.append(json_obj)
    all_gens.extend(objs)
    end_time = time.time()
    locate_time = end_time - start_time
    print(f"GENERATION TIME: {locate_time/60}")
    print("=========== end generation! =============")

save_path = f"/home/saeheeeom/data/set_consistency/data/pplm_qwen_negative.jsonl"
with open(save_path,'w') as f:
  for obj in all_gens:
    f.write(json.dumps(obj) + '\n')