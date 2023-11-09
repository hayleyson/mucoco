from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd

llama2 = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b-hf')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')



nontoxic_prompts = pd.read_json('/home/s3/hyeryung/mucoco/data/control-prompts/nontoxic_prompts-10k.jsonl',
                                lines = True)

nontoxic_prompts = nontoxic_prompts['prompt']

nontoxic_prompts['text'] = nontoxic_prompts['prompt'].apply(lambda x: x['text'])
nontoxic_prompts['toxicity'] = nontoxic_prompts['prompt'].apply(lambda x: x['toxicity'])

nontoxic_prompts = nontoxic_prompts.sort_values(by='toxicity', ascending=False)[:100]
nontoxic_prompts = nontoxic_prompts.drop(columns=['prompt', 'toxicity'])


prompt_data = Dataset.from_pandas(nontoxic_prompts)
prompt_data = prompt_data.map(lambda x: tokenizer(x['text'], padding=True, truncation=True), batched=True)
prompt_data.set_format('torch')


prompt_dataloader = DataLoader(prompt_data, batch_size=4, shuffle=True)

llama2.eval()

gen_data = {'prompt': [], 'generations': []}
with torch.no_grad():
    for batch in prompt_dataloader:
        
        continuations = llama2.generate(batch, num_return_sequences = 25, max_new_tokens=100, do_sample=True, top_p=0.95)
        print(continuations)
        break