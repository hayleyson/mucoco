"""
Evaluation 전, output 후처리 코드

태스크마다, 데이터셋마다 prompt가 평가에 들어갈지 여부가 달라질 것 같아 evaluation 단계에서 postprocessing 을 적용하지 않고, 따로 postprocessing을 적용한 파일들을 만듭니다.
참고로 prompt를 빈 문자열로 대체하는 경우, processing 이전에 같은 prompt로 묶인 generation들이 같이 묶일 수 없기 때문에 unraveled version으로 저장됩니다. 따라서 evaluation 시 source file이 필요한 경우, source file도 unraveled version을 만들어서 평가합니다. (그래서 저는 evaluate_nli.py 파일을 따로 만들어서 평가했습니다.)

기본 postprocessing (**제거): 원래 존재하던 directory에, 같은 파일명 끝에 "_"가 suffix로 붙어서 저장
nli용 postprocessing (**제거, completion text 안에서 exact match 제거): 새로운 directory에 동일한 파일명으로 저장
fluency용 postprocessing (**제거, exact match 존재하거나 첫 3-gram 일치하는 경우 prompt 빈 문자열로 대체): 새로운 directory에 동일한 파일명으로 저장
"""
import argparse
import os
import json

def replace_ast(input_dir, suffix):
    # Ensure the save directory exists

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(suffix):
            input_path = os.path.join(input_dir, filename)
            save_path = input_path + '_'

            processed_lines = []

            # Read and process each line of the JSONL file
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data = json.loads(line)

                    prompt = data.get('prompt', {}).get('text', '')
                    generations = data.get('generations', [])

                    processed_generations = []

                    for gen in generations:
                        gen_text = gen.get('text', '')

                        # Step 1: Remove all '**' from generation text
                        gen_text = gen_text.replace('**', '')

                    

                        # Append processed generation
                        processed_generations.append({"text": gen_text})

                    # Format the processed line
                    processed_lines.append({
                        "prompt": {"text": prompt},
                        "generations": processed_generations
                    })

            # Save the processed lines to the save directory
            with open(save_path, 'w', encoding='utf-8') as outfile:
                for processed_line in processed_lines:
                    outfile.write(json.dumps(processed_line) + "\n")
                    
def remove_em(input_dir, save_dir, suffix):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(suffix):
            input_path = os.path.join(input_dir, filename)
            save_path = os.path.join(save_dir, filename)

            processed_lines = []

            print(input_path)
            # Read and process each line of the JSONL file
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data = json.loads(line)

                    prompt = data.get('prompt', {}).get('text', '').lower()
                    generations = data.get('generations', [])


                    for gen in generations:
                        processed_generations = []
                        gen_text = gen.get('text', '').lower()

                        # Step 1: Remove all '**' from generation text
                        gen_text = gen_text.replace('**', '')

                        # Step 2: Replace prompt with an empty string if they are identical
                        if prompt in gen_text:#gen_text.startswith(prompt):
                            gen_to_save = gen_text.replace(prompt, "")
                        else:
                            gen_to_save = gen_text

                        # Append processed generation
                        processed_generations.append({"text": gen_to_save})

                        # Format the processed line
                        processed_lines.append({
                            "prompt": {"text": prompt},
                            "generations": processed_generations
                        })

            # Save the processed lines to the save directory
            with open(save_path, 'w', encoding='utf-8') as outfile:
                for processed_line in processed_lines:
                    outfile.write(json.dumps(processed_line) + "\n")

# suffix could be ".jsonl"

def process_jsonl_total_files(input_dir, save_dir, suffix):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(suffix):
            input_path = os.path.join(input_dir, filename)
            save_path = os.path.join(save_dir, filename)

            processed_lines = []

            # Read and process each line of the JSONL file
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data = json.loads(line)

                    prompt = data.get('prompt', {}).get('text', '')
                    generations = data.get('generations', [])


                    for gen in generations:
                        processed_generations = []
                        gen_text = gen.get('text', '')

                        # Step 1: Remove all '**' from generation text
                        gen_text = gen_text.replace('**', '')

                        # Step 2: Replace prompt with an empty string if they are identical
                        if prompt.lower() in gen_text.lower():
                            prompt_to_save = ''
                        else:
                            gens = gen_text.lower().split()
                            prmpts = prompt.lower().split()
                            if len(prmpts) > 2 and len(gens) >2:
                                if gens[0] == prmpts[0] and gens[1] == prmpts[1] and gens[2] == prmpts[2]:
                                    prompt_to_save = ''
                                else:
                                    prompt_to_save = prompt
                            else:
                                prompt_to_save = prompt

                        # Append processed generation
                        processed_generations.append({"text": gen_text})

                        # Format the processed line
                        processed_lines.append({
                            "prompt": {"text": prompt_to_save},
                            "generations": processed_generations
                        })

            # Save the processed lines to the save directory
            with open(save_path, 'w', encoding='utf-8') as outfile:
                for processed_line in processed_lines:
                    outfile.write(json.dumps(processed_line) + "\n")
                    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--option', choices=["default", "for_nli_metric", "for_fluency_metric"], required=True)
    
    args = parser.parse_args()
    
    if args.option == "default":
        replace_ast(args.input_dir, args.save_dir, args.suffix)
    elif args.option == "for_nli_metric":
        remove_em(args.input_dir, args.save_dir, args.suffix)
    elif args.option == "for_fluency_metric":
        process_jsonl_total_files(args.input_dir, args.save_dir, args.suffix)