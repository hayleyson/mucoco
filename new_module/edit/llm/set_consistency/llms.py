import os, re
from transformers import AutoTokenizer

from new_module.edit.llm.set_consistency.prompts import EDIT_PROMPT, EDIT_WITH_LOCATE_PROMPT

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

    def __init__(self, model_id: str, dataset_name: str,tensor_parallel_size: int=1):
        
        from vllm import LLM, SamplingParams
        
        self.model_id = model_id
        self.dataset_name = dataset_name
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
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
            top_p=1e-10
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)


        self.edit_prompt = EDIT_PROMPT
        self.edit_with_locate_prompt = EDIT_WITH_LOCATE_PROMPT

    def generate_batch(self, prompts: list) -> tuple:
        """
        Runs batch generation for a list of prompts.
        Returns a tuple of (responses, reasoning_tokens_list, total_tokens_list).
        """
        # apply chat template
        print(f"Prompt example before applying chat template : {prompts[0]}")

        prompts = [[ {"role": "user", 
                      "content": p}] for p in prompts]
        prompts = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]

        print(f"Prompt example after applying chat template : {prompts[0]}")

        outputs = self.model.generate(prompts, self.sampling_params)
        
        responses = []
        reasoning_tokens_list = []
        total_tokens_list = []
        
        for o in outputs:
            out = o.outputs[0]
            responses.append(out.text.strip())
            
            # Total generated tokens
            total_tokens = len(out.token_ids)
            total_tokens_list.append(total_tokens)
            
            # Extract reasoning tokens safely (if the model/vLLM version supports it)
            reasoning_tokens = 0
            if hasattr(out, 'reasoning_token_ids') and out.reasoning_token_ids is not None:
                reasoning_tokens = len(out.reasoning_token_ids)
            reasoning_tokens_list.append(reasoning_tokens)

            # print(f"Reasoning tokens: {reasoning_tokens}")
            # print(f"Total generated tokens: {total_tokens}")
            
        return responses, reasoning_tokens_list, total_tokens_list

    def set_prompt(self, data: list, located_indexes: list=None):
        
        input_text = ""
        for j, pair in enumerate(data):
            input_text += f"({j+1}) {pair}"

        if located_indexes is not None:
            return self.edit_with_locate_prompt.format(input_text=input_text, locate_labels=located_indexes, datapoint_type=self.datapoint_type, datapoint_type_capitalized=self.datapoint_type.capitalize())
        else:
            return self.edit_prompt.format(input_text=input_text, datapoint_type=self.datapoint_type)

    def parse_pairs_response(self, response: str) -> list:
        
        # Parse the output text to extract the edited question-answer pairs.
        if ('<think>' in response) and ('</think>' not in response):
            # The response got truncated before the reasoning completed.
            # In this case, we cannot extract the edited pairs.
            return []
        
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
    
    def parse_sentences_response(self, response: str) -> list:
        
        # Parse the output text to extract the edited question-answer pairs.
        if ('<think>' in response) and ('</think>' not in response):
            # The response got truncated before the reasoning completed.
            # In this case, we cannot extract the edited pairs.
            return []
        
        response = response.split('</think>')[-1].strip()
        sentences = re.split(r'\((\d+)\)\s*', response)
        
        parsed_sentences = []
        for s in sentences:
            if s == '' or s.isdigit():
                continue
            parsed_sentences.append(s.strip())
        return parsed_sentences
        
    def edit(self, data_list: list, located_indexes_list: list=None) -> str:
        
        if located_indexes_list is not None:
            prompts = [self.set_prompt(data, located_indexes) for data, located_indexes in zip(data_list, located_indexes_list)]
        else:
            prompts = [self.set_prompt(data) for data in data_list]
        responses, r_toks, t_toks = self.generate_batch(prompts)
        parsed_pairs = [self.parse_response(response) for response in responses]

        return {"edited_pairs_list": parsed_pairs, 
                "raw_response_list": responses, 
                "total_reasoning_tokens": sum(r_toks), 
                "total_completion_tokens": sum(t_toks)}

