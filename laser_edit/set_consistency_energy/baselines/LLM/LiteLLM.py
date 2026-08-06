import json, os, random, re, warnings, dotenv, torch
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import litellm
except ImportError as exc:
    warnings.warn(
        f"litellm is not installed ({exc!r}); LitellmLLM will fail if used.",
        stacklevel=2,
    )
    litellm = None  # type: ignore[misc, assignment]
    
dotenv.load_dotenv()

SKIML_API_KEY = os.getenv("SKIML_API_KEY")
SKIML_BASE_URL = os.getenv("SKIML_BASE_URL")

try:
    litellm.api_key    = os.getenv("SKIML_API_KEY")
    litellm.api_base   = os.getenv("SKIML_BASE_URL")
    litellm.ssl_verify = False
except:
    pass
        
class LitellmLLM():

    def __init__(self, params):
        
        self.params = params
        if self.params['dataset'] == 'lconvqa':
            self.datapoint_type = "question-answer pair"
        elif self.params['dataset'] == 'set_nli':
            self.datapoint_type = 'sentence'
        self.few_shot_prompts_path = os.path.join(Path(__file__).resolve().parent, "few_shot_prompts.json")
        self.shot_num = self.params['baseline']['shot_num']
        self.prompt_template_for_prediction = None
        self.prompt_template_for_locate = None
        self.prediction_type = self.params['baseline']['prediction_type']
        if not 'do_not_initialize' in self.params['baseline']:
            self.initialize_prompt()
        self.reasoning_effort = self.params['baseline']['reasoning_effort']
        
        self.model_id = params['baseline']['model']
        print(f"model: {self.model_id}")

        private_key = os.getenv('OPENAI_API_KEY')
        if not private_key:
            raise ValueError("OPENAI_API_KEY is not set")

        if "anthropic/" in self.model_id:
            self.generate = self.generate_anthropic
        elif "gemini/" in self.model_id:
            self.generate = self.generate_gemini
        elif "openai/" in self.model_id:
            self.generate = self.generate_openai
        else:
            raise ValueError(f"Model {self.model_id} not supported")

    @staticmethod
    def _usage_value(usage: Any, key: str, default: Any=0) -> Any:
        if usage is None:
            return default
        if isinstance(usage, dict):
            return usage.get(key, default) or default
        return getattr(usage, key, default) or default

    def _token_usage(self, response: Any) -> Tuple[int, int]:
        usage = getattr(response, "usage", None)
        completion_tokens = self._usage_value(usage, "completion_tokens")
        token_details = self._usage_value(usage, "completion_tokens_details", None)
        reasoning_tokens = self._usage_value(token_details, "reasoning_tokens")

        return reasoning_tokens, completion_tokens

    def _completion_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs
    
    def api_call(self, prompts: str) -> Tuple[List[str], int, int]:
        if "o1" in self.model_id:
            messages=[
            # {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompts}]
        else:
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompts}]
            
        responses, reasoning_tokens, completion_tokens = self.generate(messages)
        return responses[0], reasoning_tokens, completion_tokens
    
    def generate_openai(self, 
                 messages: List[Dict], 
                 max_new_tokens: int=4096, 
                 n: int=1,
                 top_p: float=0.96,
                 top_k: int=50,
                 temperature: float=1.0) -> Tuple[List[str], int, int]:

        responses = []
        reasoning_tokens = 0
        completion_tokens = 0
        iterations = (n + 7) // 8

        for i in range(iterations):
            remaining = n - 8 * i
            n_ = min(8, remaining)
            response = litellm.completion(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                n=n_,
                custom_llm_provider="openai",
                **self._completion_kwargs(),
            )
            r_tok, c_tok = self._token_usage(response)
            reasoning_tokens += r_tok
            completion_tokens += c_tok
            responses.extend([(c.message.content or "") for c in response.choices])
        return responses, reasoning_tokens, completion_tokens

    def generate_anthropic(self, 
                 messages: List[Dict], 
                 max_new_tokens: int=4096, 
                 n: int=1,
                 top_p: float=0.96,
                 top_k: int=50,
                 temperature: float=1.0) -> Tuple[List[str], int, int]:
        
        responses = []
        reasoning_tokens = 0
        completion_tokens = 0
        for _ in range(n):
            response = litellm.completion(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                custom_llm_provider="openai",
                extra_body={
                        "cache": {
                            "no-cache": True  # Skip cache check, get fresh response
                        }
                    },
                **self._completion_kwargs(),
            )
            r_tok, c_tok = self._token_usage(response)
            reasoning_tokens += r_tok
            completion_tokens += c_tok
            print(f"Sample response: {response}")
            responses.append(response.choices[0].message.content or "")
        return responses, reasoning_tokens, completion_tokens

    def generate_gemini(self, 
                 messages: List[Dict], 
                 max_new_tokens: int=4096, 
                 n: int=1,
                 top_p: float=0.96,
                 top_k: int=50,
                 temperature: float=1.0) -> Tuple[List[str], int, int]:
        
        responses = []
        reasoning_tokens = 0
        completion_tokens = 0
        iterations = (n + 7) // 8

        for i in range(iterations):
            remaining = n - 8 * i
            n_ = min(8, remaining)
            response = litellm.completion(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                n=n_,
                custom_llm_provider="openai",
                **self._completion_kwargs(),
            )
            r_tok, c_tok = self._token_usage(response)
            reasoning_tokens += r_tok
            completion_tokens += c_tok
            print(f"Sample response: {response}")
            _responses = [c.message.content or "" for c in response.choices]
            c0 = _responses[0]
            for _response in _responses[1:]:
                c0 = c0.replace(_response, "")
            _responses[0] = c0
            responses.extend(_responses)
        return responses, reasoning_tokens, completion_tokens

    def predict(self, pair):     
        if self.prediction_type == 'all_in_one':
            return self.predict_all_in_one(pair)
        
        else:
            raise ValueError(f"Prediction type {self.prediction_type} not supported")
        
    def predict_all_in_one(self, pair: List):
        assert len(pair) == 1

        prompts = self.finalize_prompt(pair[0], mode = 'predict')
        # print("prompt:")
        # print(prompts)
        # print()
        
        pred, r_tok, t_tok = self.api_call(prompts)
        # print("api_call result:\n",pred)

        # transform to the integer
        pred = self.post_process_prediction(pred)
        if type(pred) == int:
            pred = [pred]

        # print("prediction result:", pred)
        return pred
    
    
    def locate(self, pair):
        prompts = self.finalize_prompt(pair[0], mode = 'locate')
        # print("prompt:")
        # print(prompts)
        # print()
        
        try:
            pred_str, r_tok, t_tok = self.api_call(prompts)
        except litellm.exceptions.BadRequestError:
            print("=========================================")
            print(f"!!!Bad request error for prompts: \n{prompts}")
            print("=========================================")
            pred_str = ""
        # print(pred_str)
        # raise

        # transform to the integer
        pred_list = self.post_process_locate(pred_str)
        # print(pred_list)
        # raise
        if random.random() < 0.1:
            print("prompts:")
            print(prompts)
            print("pred_str:")
            print(pred_str)
            print("pred_list:")
            print(pred_list)
            
        return {"pred_list": pred_list,
                "pred_str": pred_str,
                "reasoning_tokens": r_tok, 
                "total_generated_tokens": t_tok}

    

    def initialize_prompt(self):
        """
        Generate a prompt-template used for prediction.

        Our prompt is composed of following four parts:
            (1) front_prompt: explain the user's intention
            (2) (optional) few_shot examples
            (3) input-output pairs (= what we want to evaluate)
            (4) end_prompt: 
                For prediction  : "Consistency: "
                For locate      : "Inconsistent examples: "

        This function aims to complete (1) and (2).
        """
        
        # (1) front_prompt
        tasks = ['prediction', 'locate']
        # tasks = ['prediction']
        self.prompt_template_for_prediction = f"Tell me whether the following {self.datapoint_type}s are consistent or inconsistent. \n"
        self.prompt_template_for_locate = f"Find the {self.datapoint_type}s among the following that are logically inconsistent with the rest. Specifically, identify the minimal collection of inconsistent {self.datapoint_type}s such that the remaining {self.datapoint_type}s are logically consistent with one another. If there are no inconsistent {self.datapoint_type}s, return nothing."
        self.prompt_template_for_prediction_and_locate = f"Your goal is to solve the following two tasks.\n First, {self.prompt_template_for_prediction} Second, {self.prompt_template_for_locate}"
        
        if self.shot_num == 0:
            return
        
        # (2) few-shot examples
        with open(self.few_shot_prompts_path, 'r') as f:
            prompt_dict = json.load(f)
        
        prompt_dicts_for_task = prompt_dict[self.params['dataset']]

        for t in tasks:
            prompt_dict_for_t= prompt_dicts_for_task[t][self.prediction_type] 
            prompt_list_for_t = [val for key, val in prompt_dict_for_t.items() if int(key) <= self.shot_num]
            if len(prompt_list_for_t) < self.shot_num:
                print(f"Number of few_shot examples is less than you want for task {t}.")
                print(f"Currently we have {len(prompt_list_for_t)}, while you want {self.shot_num}.")
                print(f"Nontheless, we don't raise any error. We utilize {len(prompt_list_for_t)} number of few-shot examples")
                self.shot_num = len(prompt_list_for_t)
                self.params['baseline']['shot_num'] = len(prompt_list_for_t)

            for i, prompt in enumerate(prompt_list_for_t):
                setattr(self, 
                        f"prompt_template_for_{t}", 
                        getattr(self, f"prompt_template_for_{t}") + f"\n[example {i+1}]\n{prompt}\n")
                # for example, if t == 'prediction', then this code executes:
                #   self['prompt_template_for_prediction'] += f"\n[example {i+1}]\n{prompt}\n"


    def finalize_prompt(self, pairs: List, mode: str) -> str:
        prompts = ""
        if mode == 'predict':
            front_prompt = self.prompt_template_for_prediction
            if self.prediction_type == 'many_to_one':
                front_prompt += f"\nPlease let me know that whether the {self.datapoint_type}s in premise are logically consistent or inconsistent with the {self.datapoint_type} in hypothesis. Furthermore, if the {self.datapoint_type}s in premise is already logically inconsistent, then your answer should be inconsistent.\n"
            end_prompt = f"provide your consistency judgment by choosing either 'consistent' or 'inconsistent' After the 'Consistency:' mark"

            tmp = front_prompt + "\n [Problem]\n"
            if self.prediction_type in {"all_in_one", "one_to_one"}:
                for i, p in enumerate(pairs[0]):
                    tmp += f"({i+1}) {p}"
                tmp += " \n "
                tmp += end_prompt
                prompts = tmp
            elif self.prediction_type == 'many_to_one':
                tmp += "\n[Premise] "
                for i, premise in enumerate(pairs['premise']):
                    tmp += f"({i+1}) {premise}"
                    tmp += " \n "
                # print("tmp:", tmp)
                tmp += "[Hypothesis] "
                tmp += f"({len(pairs['premise'])+1}) {pairs['hypothesis']}\n"
                tmp += end_prompt
                prompts = tmp

        elif mode == 'locate':
            front_prompt = self.prompt_template_for_locate
            end_prompt = f"Your response should only contain the numbers of the inconsistent {self.datapoint_type}s. \nInconsistent {self.datapoint_type}s:"
            
            tmp = front_prompt + "\n [Problem]\n"
            for i, p in enumerate(pairs[0]):
                tmp += f"({i+1}) {p}"
            tmp += " \n "
            tmp += end_prompt
            prompts = tmp


        # elif mode == 'prediction_and_locate':
        #     front_prompt = self.prompt_template_for_prediction_and_locate
        #     end_prompt = "You should give me two answers. For the first task, your answer should be only one word, either 'consistent' or 'inconsistent'.\nFor the second task, your answer should be only the number of inconsistent pair from the rest. \n\n Consistency: \n\n Inconsistent pairs: "

        else:
            print(f"Invalid mode here. Your mode is {mode}.")
            raise NotImplementedError
        
        # final pre-processing to avoid JSON parsing error.
        prompts = prompts.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'").replace('—', "-")
        # print("prompts:\n",prompts)
        # raise
        return prompts
    
    
    def post_process_prediction(self, prediction):
        # Not completed yet.

        prediction = prediction[prediction.rfind("Consistency:") + len("Consistency:"):]

        incon_detect = False
        con_detect = True
        if 'inconsistent' in prediction:
            incon_detect = True
        if ('consistent' in prediction):
            con_detect = True
            
        # if random.random() < 0.1:
            # print(f"prediction:", prediction)
            # print(f"con_detect, incon_detect: {con_detect}, {incon_detect}")
        if con_detect and (not incon_detect):
            return 0 # 0 means consistent
        elif incon_detect:
            return 1 # 1 means inconsistent
        else:
            return int(torch.randint(0, 2, (1,)))

    def post_process_locate(self, response_text):
        identifier = "Inconsistent pairs:"
        if '\n' in response_text:
            response_text = response_text.split('\n')[0]
        if identifier in response_text:
            text_index = response_text.find(identifier)
            text = response_text[text_index + len(identifier):]
        else:
            text = response_text
        patterns = re.findall(r'(\d{1,2})', text)
        patterns = [int(p) for p in patterns]
        return patterns