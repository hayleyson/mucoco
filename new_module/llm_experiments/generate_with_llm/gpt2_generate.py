import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed 
from tqdm import tqdm

set_seed(200)

prompts = [
    "The mountain", "The dream", "The secret", "The stranger", "The child",
    "The machine", "The morning", "The storm", "The journey", "The answer",
    "The letter", "The memory", "The door", "The garden", "It is the beginning"
]

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("gpt2-large")

# GPT-2 has no pad token; use EOS as pad to avoid warnings/errors
pad_id = tokenizer.eos_token_id

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

seq_lengths = [12, 20, 50]
generated_results = []

for seq_length in seq_lengths:
    print('----- doing seq_length ', seq_length, ' -----')
    for prompt in tqdm(prompts):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate continuation
        gen_ids = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.96,
            max_new_tokens=seq_length,
            pad_token_id=pad_id,
            num_return_sequences=20,
        )
        

        # Decode; by default this returns the full text (prompt + continuation)
        full_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        generated_result = {
            "prompt": {"text": prompt},
            "generations": [{"text": full_text[len(prompt):]} for full_text in full_texts],
            "seq_length": seq_length,  # singular name is clearer
        }
        
        generated_results.append(generated_result)

with open('/data/hyeryung/mucoco/new_module/data/sentiment/new_dev_set.jsonl', 'w') as f:
    for line in generated_results:
        json.dump(line, f)
        f.write('\n')
