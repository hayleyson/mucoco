import argparse, json, os
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--model_name', type=str, default='gpt2-large')
parser.add_argument('--file_type', type=str, required=False, default='jsonl', choices=['jsonl', 'txt'])
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

num_tokens = 0
if args.file_type == 'jsonl':
    with open(args.input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            generations = data['generations']
            for gen in generations:
                tokens = tokenizer.encode(gen['text'])
                num_tokens += len(tokens)
elif args.file_type == 'txt':
    with open(args.input_file, 'r') as f:
        for line in f:
            tokens = tokenizer.encode(line)
            num_tokens += len(tokens)
else:
    raise ValueError(f"Invalid file type: {args.file_type}")

print(num_tokens)

input_file_name, input_file_ext = os.path.splitext(args.input_file)
with open(input_file_name + '.num_tokens', 'w') as f:
    f.write(str(num_tokens) + '\n')