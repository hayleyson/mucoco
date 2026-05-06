import os, json, argparse, time
from pathlib import Path
from new_module.llm_experiments.generate_with_llm.prompts import get_prompt
from new_module.llm_experiments.generate_with_llm.llm import LitellmLLM, OpenAILLM, HuggingfaceLLM, AnthropicLLM, GoogleLLM, VllmLLM
from new_module.data.nli_ifeval.ifeval_prompts import get_ifeval_prompt

ROOT_DIR = Path(__file__).resolve().parent

with open(ROOT_DIR / "few_shot_prompts.json", "r") as f:
    few_shot_prompts = json.load(f)


def _format_few_shot_nli(entry: dict) -> str:
    return f"Premise: {entry['premise']}\nHypothesis: {entry['hypothesis']}"


def _format_few_shot_nontoxic(entry: dict) -> str:
    return f"Prompt: {entry['prompt']}\nContinuation: {entry['continuation']}"


def _format_few_shot_nli_nontoxic(entry: dict) -> str:
    return f"Text: {entry['text']}\nReply: {entry['reply']}"

def _format_few_shot_rewrite_hypothesis_toxic(entry: dict) -> str:
    return f"Premise: {entry['premise']}\nHypothesis: {entry['hypothesis']}\nRewritten Toxic Hypothesis: {entry['rewritten_hypothesis']}"

def get_few_shot_prompts(prompt_type: str, num_shots: int = 5):
    if "nli_nontoxic" in prompt_type or "nli+nontoxic" in prompt_type:
        key = "nli+nontoxic"
        fmt = _format_few_shot_nli_nontoxic
    elif "nli" in prompt_type:
        key = "nli"
        fmt = _format_few_shot_nli
    elif "nontoxic" in prompt_type:
        key = "nontoxic"
        fmt = _format_few_shot_nontoxic
    elif "rewrite_hypothesis_toxic" in prompt_type:
        key = "rewrite_hypothesis_toxic"
        fmt = _format_few_shot_rewrite_hypothesis_toxic
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    few_shots = []
    for i in range(num_shots):
        entry = few_shot_prompts[key][str(i + 1)]
        few_shots.append(fmt(entry))
    return "\n\n".join(few_shots)


def build_messages_for_prompt(
    prompt_type: str,
    user_prompt_template: str,
    system_prompt: str,
    prompt_text: str,
    additional_text: str = None,
    num_shots: int =0,
):
    """Build chat ``messages`` for one row.

    Returns ``(messages, user_content, constraint_ids)``. ``constraint_ids`` is only
    set for ``nli_ifeval``; otherwise ``None``.
    """
    if "few_shot" in prompt_type and num_shots == 0:
        raise ValueError(f"num_shots must be greater than 0 for {prompt_type}")
    
    kwargs = {}
    if "comment" in prompt_type:
        kwargs = {
            "article_excerpt": additional_text,
            "comment_prefix": prompt_text
        }
    elif "rewrite_hypothesis" in prompt_type:
        kwargs = {
            "premise": prompt_text,
            "hypothesis": additional_text}
    else:
        kwargs = {
            "prompt": prompt_text
        }
    if "few_shot" in prompt_type:
        kwargs |= {
            "examples": get_few_shot_prompts(prompt_type, num_shots)
        }
    
    user_content = user_prompt_template.format(**kwargs)
    
    if prompt_type == "nli_ifeval":
        result = get_ifeval_prompt()
        user_content = (
            user_content
            + "\n"
            + result["instruction"]
        )
        constraint_ids = result["instruction_id_list"]
    else:
        constraint_ids = None 
        
    if system_prompt != "":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]
    return messages, user_content, constraint_ids


def generate_and_save_result(args):
    
    if args.model_access_method == "litellm":
        llm = LitellmLLM(args.model)
    elif args.model_access_method == "vllm":
        llm = VllmLLM(args.model)
    elif args.model_access_method == "openai":
        llm = OpenAILLM(args.model)
    elif args.model_access_method == "anthropic":
        llm = AnthropicLLM(args.model)
    elif args.model_access_method == "google":
        llm = GoogleLLM(args.model)
    elif args.model_access_method == "hf":
        llm = HuggingfaceLLM(args.model)
    else:
        raise ValueError(f"Invalid model access method: {args.model_access_method}")
    
    with open(args.input_file_path,'r') as f:
        raw_data = f.readlines()
    if "comment" in args.prompt_type:
        additional_texts = [json.loads(line)['excerpt'] for line in raw_data]
        prompts = [json.loads(line)['comment_prefix'] for line in raw_data]
    elif "rewrite_hypothesis" in args.prompt_type:
        prompts = [json.loads(line)['premise'] for line in raw_data]
        additional_texts = [json.loads(line)['hypothesis'] for line in raw_data]
    elif args.input_file_path.endswith('.jsonl'): ##toxic,senti,nli
        additional_texts = None
        prompts = [json.loads(line)['prompt']['text'] for line in raw_data]
    else:## txt file ##formality transfer
        additional_texts = None
        prompts = [line.rstrip() for line in raw_data]

    system_prompt, user_prompt_template = get_prompt(args)
    print(f"System_prompt: {system_prompt}")
    print(f"User_prompt_template: {user_prompt_template}")
    print('-'*50)
    
    ## generate responses
    max_prompt_count = len(prompts) if args.num_test_prompts == -1 else args.num_test_prompts
    slice_range = slice(args.prompt_start_index, args.prompt_start_index + max_prompt_count)
    prompts = prompts[slice_range]
    if additional_texts is not None:
        additional_texts = additional_texts[slice_range]
    start_time = time.time()
    
    with open(args.file_save_path, 'w') as f:
        if args.model_access_method == "vllm":
            messages_list = []
            user_contents = []
            constraint_ids_list = []
            for i, p in enumerate(prompts):
                messages, user_content, constraint_ids = build_messages_for_prompt(
                    args.prompt_type,
                    user_prompt_template,
                    system_prompt,
                    p,
                    additional_texts[i] if additional_texts is not None else None,
                    args.num_shots,
                )
                if i == 0:
                    print(f"Sample messages: {messages}")
                    print(f"Sample user_content: {user_content}")
                messages_list.append(messages)
                user_contents.append(user_content)
                constraint_ids_list.append(constraint_ids)
            responses_list = llm.generate_batch(
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                n=args.num_return_sequences,
                messages_list=messages_list
            )
            for i, (p, user_content, constraint_ids, responses) in enumerate(
                zip(prompts, user_contents, constraint_ids_list, responses_list)
            ):
                if args.prompt_type == "nli_ifeval":
                    formatted_generated_text = {'prompt': {'text': p, 'full_prompt': user_content, 'instruction_id_list': constraint_ids},
                                            'generations': [{'text': o} for o in responses]}
                elif "comment" in args.prompt_type:
                    formatted_generated_text = {'prompt': {'text': p, 'article_excerpt': additional_texts[i]},
                                            'generations': [{'text': o} for o in responses]}
                elif "rewrite_hypothesis" in args.prompt_type:
                    formatted_generated_text = {'prompt': {'text': p, 'original_hypothesis': additional_texts[i]},
                                            'generations': [{'text': o} for o in responses]}
                else:
                    formatted_generated_text = {'prompt': {'text': p},
                                            'generations': [{'text': o} for o in responses]}
                f.write(json.dumps(formatted_generated_text) + '\n')
                f.flush()

        else:
            for i, p in enumerate(prompts):
                messages, user_content, constraint_ids = build_messages_for_prompt(
                    args.prompt_type,
                    user_prompt_template,
                    system_prompt,
                    p,
                    additional_texts[i] if additional_texts is not None else None,
                    args.num_shots,
                )
                if i == 0:
                    print(f"Sample messages: {messages}")
                    print(f"Sample user_content: {user_content}")
                    
                responses = llm.generate(
                    top_p=args.top_p,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    n=args.num_return_sequences,
                    messages=messages
                )
                
                # break
                if args.prompt_type == "nli_ifeval":
                    formatted_generated_text = {'prompt': {'text': p, 'full_prompt': user_content, 'instruction_id_list': constraint_ids},
                                            'generations': [{'text': o} for o in responses]}
                elif "comment" in args.prompt_type:
                    formatted_generated_text = {'prompt': {'text': p, 'article_excerpt': additional_texts[i]},
                                            'generations': [{'text': o} for o in responses]}
                elif "rewrite_hypothesis" in args.prompt_type:
                    formatted_generated_text = {'prompt': {'text': p, 'original_hypothesis': additional_texts[i]},
                                            'generations': [{'text': o} for o in responses]}
                else:
                    formatted_generated_text = {'prompt': {'text': p},
                                            'generations': [{'text': o} for o in responses]}
                f.write(json.dumps(formatted_generated_text) + '\n')
                f.flush() 
    end_time = time.time()
    
    with open(args.file_save_path.replace('.jsonl', '.time'), 'w') as f:
        f.write(str(end_time - start_time) + '\n')
    print(f"Total time taken (seconds): {end_time - start_time}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--file_save_path', type=str)
    parser.add_argument('--input_file_path', type=str)
    parser.add_argument('--prompt_type', type=str)    
    parser.add_argument('--num_shots', type=int, default=0)
    parser.add_argument('--num_return_sequences', type=int, default=10)    
    parser.add_argument('--prompt_start_index', type=int, default=0)
    parser.add_argument('--num_test_prompts', type=int, default=-1)
    parser.add_argument('--model_access_method', type=str, default="hf")
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    args = parser.parse_args()
    # for backward compatibility
    args.max_tokens = args.max_new_tokens
    
    print(f"Arguments: {args}")
    print('-'*50)
    
    save_dir = os.path.dirname(args.file_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    generate_and_save_result(args)
    
    
    