import os, dotenv, random, torch, warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import litellm
except ImportError as exc:
    warnings.warn(
        f"litellm is not installed ({exc!r}); LitellmLLM will fail if used.",
        stacklevel=2,
    )
    litellm = None  # type: ignore[misc, assignment]
    
try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams
except ImportError as exc:
    warnings.warn(
        f"vllm is not installed ({exc!r}); VllmLLM will fail if used.",
        stacklevel=2,
    )
    VLLMEngine = None  # type: ignore[misc, assignment]
    SamplingParams = None  # type: ignore[misc, assignment]

dotenv.load_dotenv()

SKIML_API_KEY = os.getenv("SKIML_API_KEY")
SKIML_BASE_URL = os.getenv("SKIML_BASE_URL")

try:
    litellm.api_key    = os.getenv("SKIML_API_KEY")
    litellm.api_base   = os.getenv("SKIML_BASE_URL")
    litellm.ssl_verify = False
except:
    pass

def set_global_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
class LitellmLLM(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name)
        if "anthropic/" in model_name:
            self.generate = self.generate_anthropic
        elif "gemini/" in model_name:
            self.generate = self.generate_gemini
        elif "openai/" in model_name:
            self.generate = self.generate_openai
        else:
            raise ValueError(f"Model {model_name} not supported")
        
    def generate_openai(self, 
                 messages: List[Dict], 
                 max_new_tokens: int=4096, 
                 n: int=1,
                 top_p: float=0.96,
                 top_k: int=50,
                 temperature: float=1.0) -> List[str]:

        responses = []
        iterations = (n + 7) // 8

        for i in range(iterations):
            remaining = n - 8 * i
            n_ = min(8, remaining)
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                n=n_,
                custom_llm_provider="openai",
            )
            responses.extend([(c.message.content or "") for c in response.choices])
        return responses

    def generate_anthropic(self, 
                 messages: List[Dict], 
                 max_new_tokens: int=4096, 
                 n: int=1,
                 top_p: float=0.96,
                 top_k: int=50,
                 temperature: float=1.0) -> List[str]:
        
        responses = []
        for _ in range(n):
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                custom_llm_provider="openai",
                extra_body={
                        "cache": {
                            "no-cache": True  # Skip cache check, get fresh response
                        }
                    }
            )
            print(f"Sample response: {response}")
            responses.append(response.choices[0].message.content or "")
        return responses

    def generate_gemini(self, 
                 messages: List[Dict], 
                 max_new_tokens: int=4096, 
                 n: int=1,
                 top_p: float=0.96,
                 top_k: int=50,
                 temperature: float=1.0) -> List[str]:
        
        responses = []
        iterations = (n + 7) // 8

        for i in range(iterations):
            remaining = n - 8 * i
            n_ = min(8, remaining)
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                n=n_,
                custom_llm_provider="openai",
            )
            print(f"Sample response: {response}")
            _responses = [c.message.content or "" for c in response.choices]
            c0 = _responses[0]
            for _response in _responses[1:]:
                c0 = c0.replace(_response, "")
            _responses[0] = c0
            responses.extend(_responses)
        return responses
