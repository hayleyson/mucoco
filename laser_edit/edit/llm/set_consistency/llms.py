import os, re
from transformers import AutoTokenizer

from laser_edit.edit.llm.set_consistency.prompts import EDIT_PROMPT, EDIT_WITH_LOCATE_PROMPT


def is_qwen3_model(model_id: str) -> bool:
    return "qwen3" in model_id.lower()


def qwen3_chat_template_kwargs(model_id: str) -> dict:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if is_qwen3_model(model_id):
        kwargs["enable_thinking"] = True
    return kwargs


def greedy_generation_kwargs() -> dict:
    """Deterministic decoding."""
    return {"do_sample": False}


def nucleus_generation_kwargs() -> dict:
    """Match toxicity/NLI Qwen2.5 editing: temp=1.0, top_p=0.96."""
    return {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.96,
    }


def qwen3_generation_kwargs(model_id: str) -> dict:
    if not is_qwen3_model(model_id):
        return nucleus_generation_kwargs()
    return {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
    }


def generation_kwargs_for_model(model_id: str, decoding: str = "auto") -> dict:
    """
    Resolve HF generate() kwargs.

    decoding:
      - auto: Qwen3 thinking-mode sampling; nucleus (temp=1.0, top_p=0.96) for others
      - nucleus: force toxicity/NLI-matched sampling (temp=1.0, top_p=0.96)
      - greedy: force greedy for any model
    """
    if decoding == "greedy":
        return greedy_generation_kwargs()
    if decoding == "nucleus":
        return nucleus_generation_kwargs()
    if decoding != "auto":
        raise ValueError(
            f"Unsupported decoding={decoding!r}; expected 'auto', 'nucleus', or 'greedy'"
        )
    return qwen3_generation_kwargs(model_id)


def qwen3_max_new_tokens(model_id: str) -> int:
    return 32768 if is_qwen3_model(model_id) else 4096


def reasoning_tokens_from_response(tokenizer, response: str) -> int:
    """
    Token-count for the reasoning segment in models that emit
    `<think>...</think>`. Matches parse_* handling:
    counts the inner span when closed; if only the opener is present (truncated),
    counts tokens after the opener.
    """
    open_tag = "<think>"
    close_tag = "</think>"
    if open_tag not in response:
        return 0
    after_open = response.split(open_tag, 1)[1]
    if close_tag in after_open:
        reasoning_text = after_open.split(close_tag, 1)[0]
    else:
        reasoning_text = after_open  # truncated before closing tag
    reasoning_text = reasoning_text.strip()
    if not reasoning_text:
        return 0
    return len(tokenizer.encode(reasoning_text, add_special_tokens=False))


class GPT():

    def __init__(self, model_id: str, reasoning_effort: str, dataset_name: str):
        
        from openai import OpenAI
        
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.dataset_name = dataset_name
        if self.dataset_name == 'lconvqa':
            self.datapoint_type = "question-answer pair"
            self.parse_response = self.parse_pairs_response
        elif self.dataset_name == 'set_nli':
            self.datapoint_type = 'sentence'
            self.parse_response = self.parse_sentences_response

        self.edit_prompt = EDIT_PROMPT
        self.edit_with_locate_prompt = EDIT_WITH_LOCATE_PROMPT

    def generate(self, prompt: str) -> tuple:

        if self.reasoning_effort is not None:
            response = self.client.chat.completions.create(
                    model=self.model_id, 
                    messages=[{
                    "role": "user",
                    "content": prompt  
                    }],
                    reasoning_effort=self.reasoning_effort
                )
        else:
            response = self.client.chat.completions.create(
                model=self.model_id, 
                messages=[{
                    "role": "user",
                    "content": prompt  
                    }],
            )

        output_text = response.choices[0].message.content
        total_generated_tokens = response.usage.completion_tokens
        
        # Extract reasoning tokens safely (defaults to 0 if the model doesn't use reasoning)
        reasoning_tokens = 0
        if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens

        return output_text, reasoning_tokens, total_generated_tokens

    def set_prompt(self, data: list, located_indexes: list= None):
        
        input_text = ""
        for j, pair in enumerate(data):
            input_text += f"({j+1}) {pair}"

        if located_indexes is not None:
            return self.edit_with_locate_prompt.format(input_text=input_text, locate_labels=located_indexes, datapoint_type=self.datapoint_type, datapoint_type_capitalized=self.datapoint_type.capitalize())
        else:
            return self.edit_prompt.format(input_text=input_text, datapoint_type=self.datapoint_type)

    def parse_pairs_response(self, response: str) -> list:
        
        # Parse the output text to extract the edited question-answer pairs.
        pairs = re.split(r'\((\d+)\)\s*', response)
        
        parsed_pairs = []
        for p in pairs:
            if p == '' or p.isdigit():
                continue
            try:
                q, a = p.split(',')
                parsed_pairs.append((q.replace('question: ', '').strip(), a.replace('answer: ', '').strip().rstrip('.'), None))
            except:
                print(f"Warning - failed to parse:\n{p}")
                parsed_pairs.append((p, None, None))
        
        return parsed_pairs
    
    def parse_sentences_response(self, response: str) -> list:
        
        # Parse the output text to extract the edited sentences.
        sentences = re.split(r'\((\d+)\)\s*', response)
        
        parsed_sentences = []
        for p in sentences:
            if p == '' or p.isdigit():
                continue
            parsed_sentences.append(p.strip())
        return parsed_sentences
        
    def edit(self, data: list, located_indexes: list=None) -> str:
        
        prompt = self.set_prompt(data, located_indexes)
        # print(f"prompt:\n {prompt}")
        response, r_tok, t_tok = self.generate(prompt)
        parsed_datapoints = self.parse_response(response)

        return {"edited_pairs": parsed_datapoints, 
                "raw_response": response, 
                "reasoning_tokens": r_tok, 
                "total_generated_tokens": t_tok}

class HFModel():

    def __init__(
        self,
        model_id: str,
        dataset_name: str,
        tensor_parallel_size: int = 1,
        decoding: str = "auto",
        seed: int = 42,
    ):
        import torch
        from transformers import AutoModelForCausalLM

        self.model_id = model_id
        self.dataset_name = dataset_name
        self.decoding = decoding
        self.seed = seed
        self.generation_kwargs = generation_kwargs_for_model(model_id, decoding=decoding)
        print(f"[HFModel] decoding={decoding!r} seed={seed} generation_kwargs={self.generation_kwargs}")
        if self.generation_kwargs.get("do_sample", False):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if self.dataset_name == 'lconvqa':
            self.datapoint_type = "question-answer pair"
            self.parse_response = self.parse_pairs_response
        elif self.dataset_name == 'set_nli':
            self.datapoint_type = 'sentence'
            self.parse_response = self.parse_sentences_response

        load_kwargs = {"trust_remote_code": True}

        if torch.cuda.is_available():
            if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = torch.float16

        # tensor_parallel_size>1: shard weights across GPUs (similar intent to vLLM TP).
        if tensor_parallel_size > 1 and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["device_map"] = {"": "cpu"}

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.edit_prompt = EDIT_PROMPT
        self.edit_with_locate_prompt = EDIT_WITH_LOCATE_PROMPT
        self.max_new_tokens = qwen3_max_new_tokens(self.model_id)

    def generate(self, prompt: str) -> tuple:
        """Generate one prompt using Qwen3 thinking-mode sampling when applicable."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, **qwen3_chat_template_kwargs(self.model_id)
        )
        inputs = self.tokenizer(formatted, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.generation_kwargs,
            )

        new_tokens = output_ids[0, prompt_len:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Reasoning tokens are filled in edit() via reasoning_tokens_from_response after parsing.
        total_generated_tokens = int(new_tokens.shape[0])
        return output_text, 0, total_generated_tokens

    def set_prompt(self, data: list, located_indexes: list=None):
        
        input_text = ""
        for j, pair in enumerate(data):
            input_text += f"({j+1}) {pair}"

        if located_indexes is not None:
            return self.edit_with_locate_prompt.format(input_text=input_text, locate_labels=located_indexes, datapoint_type=self.datapoint_type, datapoint_type_capitalized=self.datapoint_type.capitalize())
        else:
            return self.edit_prompt.format(input_text=input_text, datapoint_type=self.datapoint_type)

    def parse_pairs_response(self, response: str, fallback_data: list = None) -> list:
        
        # Parse the output text to extract the edited question-answer pairs.
        if ('<think>' in response) and ('</think>' not in response):
            # The response got truncated before the reasoning completed.
            # Keep the original set rather than replacing it with an empty edit.
            return list(fallback_data) if fallback_data is not None else []
        
        response = response.split('</think>')[-1].strip()
        pairs = re.split(r'\((\d+)\)\s*', response)
        
        parsed_pairs = []
        for p in pairs:
            if p == '' or p.isdigit():
                continue
            try:
                q, a = p.split(',')
                parsed_pairs.append((q.replace('question: ', '').strip(), a.replace('answer: ', '').strip().rstrip('.'), None))
            except:
                print(f"Warning - failed to parse:\n{p}")
                parsed_pairs.append((p, None, None))
        
        return parsed_pairs
    
    def parse_sentences_response(self, response: str, fallback_data: list = None) -> list:
        
        # Parse the output text to extract the edited question-answer pairs.
        if ('<think>' in response) and ('</think>' not in response):
            # The response got truncated before the reasoning completed.
            # Keep the original set rather than replacing it with an empty edit.
            return list(fallback_data) if fallback_data is not None else []
        
        response = response.split('</think>')[-1].strip()
        sentences = re.split(r'\((\d+)\)\s*', response)
        
        parsed_sentences = []
        for s in sentences:
            if s == '' or s.isdigit():
                continue
            parsed_sentences.append(s.strip())
        return parsed_sentences
        
    def edit(self, data_list: list, located_indexes_list: list=None):
        responses = []
        r_toks = []
        t_toks = []
        parsed_pairs = []

        if located_indexes_list is not None:
            iterator = zip(data_list, located_indexes_list)
        else:
            iterator = ((data, None) for data in data_list)

        for data, located_indexes in iterator:
            prompt = self.set_prompt(data, located_indexes)
            response, _, t_tok = self.generate(prompt)
            responses.append(response)
            parsed_pairs.append(self.parse_response(response, data))
            r_toks.append(reasoning_tokens_from_response(self.tokenizer, response))
            t_toks.append(t_tok)

        return {"edited_pairs_list": parsed_pairs,
                "raw_response_list": responses,
                "total_reasoning_tokens": sum(r_toks),
                "total_completion_tokens": sum(t_toks)}


class VllmModel():

    def __init__(
        self,
        model_id: str,
        dataset_name: str,
        tensor_parallel_size: int = 1,
        decoding: str = "auto",
        seed: int = 42,
    ):

        from vllm import LLM, SamplingParams

        self.model_id = model_id
        self.dataset_name = dataset_name
        self.decoding = decoding
        self.seed = seed
        if self.dataset_name == 'lconvqa':
            self.datapoint_type = "question-answer pair"
            self.parse_response = self.parse_pairs_response
        elif self.dataset_name == 'set_nli':
            self.datapoint_type = 'sentence'
            self.parse_response = self.parse_sentences_response

        self.model = LLM(
            model=self.model_id,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9
        )
        gen_kwargs = generation_kwargs_for_model(model_id, decoding=decoding)
        print(f"[VllmModel] decoding={decoding!r} seed={seed} generation_kwargs={gen_kwargs}")
        if gen_kwargs.get("do_sample", False):
            vllm_kwargs = {k: v for k, v in gen_kwargs.items() if k != "do_sample"}
            self.sampling_params = SamplingParams(
                max_tokens=qwen3_max_new_tokens(self.model_id),
                seed=seed,
                **vllm_kwargs,
            )
        else:
            # temperature=0 is vLLM's greedy / deterministic path
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=qwen3_max_new_tokens(self.model_id),
                seed=seed,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        self.edit_prompt = EDIT_PROMPT
        self.edit_with_locate_prompt = EDIT_WITH_LOCATE_PROMPT

    def generate_batch(self, prompts: list) -> tuple:
        """
        Runs batch generation for a list of prompts.
        Returns a tuple of (responses, reasoning_tokens_list, total_tokens_list).
        """
        print(f"Prompt example before applying chat template : {prompts[0]}")

        prompts = [[ {"role": "user",
                      "content": p}] for p in prompts]
        prompts = [
            self.tokenizer.apply_chat_template(
                p, **qwen3_chat_template_kwargs(self.model_id)
            )
            for p in prompts
        ]

        print(f"Prompt example after applying chat template : {prompts[0]}")

        outputs = self.model.generate(prompts, self.sampling_params)

        responses = []
        reasoning_tokens_list = []
        total_tokens_list = []

        for o in outputs:
            out = o.outputs[0]
            responses.append(out.text.strip())

            total_tokens = len(out.token_ids)
            total_tokens_list.append(total_tokens)

            reasoning_tokens = 0
            if hasattr(out, 'reasoning_token_ids') and out.reasoning_token_ids is not None:
                reasoning_tokens = len(out.reasoning_token_ids)
            reasoning_tokens_list.append(reasoning_tokens)

        return responses, reasoning_tokens_list, total_tokens_list

    def set_prompt(self, data: list, located_indexes: list=None):

        input_text = ""
        for j, pair in enumerate(data):
            input_text += f"({j+1}) {pair}"

        if located_indexes is not None:
            return self.edit_with_locate_prompt.format(input_text=input_text, locate_labels=located_indexes, datapoint_type=self.datapoint_type, datapoint_type_capitalized=self.datapoint_type.capitalize())
        else:
            return self.edit_prompt.format(input_text=input_text, datapoint_type=self.datapoint_type)

    def parse_pairs_response(self, response: str, fallback_data: list = None) -> list:

        # Parse the output text to extract the edited question-answer pairs.
        if ('<think>' in response) and ('</think>' not in response):
            # The response got truncated before the reasoning completed.
            # Keep the original set rather than replacing it with an empty edit.
            return list(fallback_data) if fallback_data is not None else []

        response = response.split('</think>')[-1].strip()
        pairs = re.split(r'\((\d+)\)\s*', response)

        parsed_pairs = []
        for p in pairs:
            if p == '' or p.isdigit():
                continue
            try:
                q, a = p.split(',')
                parsed_pairs.append((q.replace('question: ', '').strip(), a.replace('answer: ', '').strip().rstrip('.'), None))
            except Exception:
                print(f"Warning - failed to parse:\n{p}")
                parsed_pairs.append((p, None, None))

        return parsed_pairs

    def parse_sentences_response(self, response: str, fallback_data: list = None) -> list:

        if ('<think>' in response) and ('</think>' not in response):
            return list(fallback_data) if fallback_data is not None else []

        response = response.split('</think>')[-1].strip()
        sentences = re.split(r'\((\d+)\)\s*', response)

        parsed_sentences = []
        for s in sentences:
            if s == '' or s.isdigit():
                continue
            parsed_sentences.append(s.strip())
        return parsed_sentences

    def edit(self, data_list: list, located_indexes_list: list=None):

        if located_indexes_list is not None:
            prompts = [self.set_prompt(data, located_indexes) for data, located_indexes in zip(data_list, located_indexes_list)]
        else:
            prompts = [self.set_prompt(data) for data in data_list]
        responses, r_toks, t_toks = self.generate_batch(prompts)
        parsed_pairs = [
            self.parse_response(response, data)
            for response, data in zip(responses, data_list)
        ]

        return {"edited_pairs_list": parsed_pairs,
                "raw_response_list": responses,
                "total_reasoning_tokens": sum(r_toks),
                "total_completion_tokens": sum(t_toks)}
