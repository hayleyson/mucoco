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
    from anthropic import Anthropic
except ImportError as exc:
    warnings.warn(
        f"anthropic is not installed ({exc!r}); AnthropicLLM will fail if used.",
        stacklevel=2,
    )
    Anthropic = None  # type: ignore[misc, assignment]

try:
    from openai import OpenAI
except ImportError as exc:
    warnings.warn(
        f"openai is not installed ({exc!r}); OpenAILLM will fail if used.",
        stacklevel=2,
    )
    OpenAI = None  # type: ignore[misc, assignment]

try:
    import google.generativeai as genai
except ImportError as exc:
    warnings.warn(
        f"google.generativeai is not installed ({exc!r}); GoogleLLM will fail if used.",
        stacklevel=2,
    )
    genai = None  # type: ignore[misc, assignment]

try:
    from transformers import pipeline, AutoTokenizer
except ImportError as exc:
    warnings.warn(
        f"transformers is not installed ({exc!r}); HuggingfaceLLM and VllmLLM will fail if used.",
        stacklevel=2,
    )
    pipeline = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

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
    _SUPPORTED_MODELS = [
        "anthropic/claude-sonnet-4-6",
        "gemini/gemini-2.5-flash",
        "openai/gpt-5-mini-2025-08-07",
    ]
    
    def __init__(self, model_name: str):
        assert model_name in self._SUPPORTED_MODELS, f"Model {model_name} not supported"
        super().__init__(model_name)
        if model_name == "anthropic/claude-sonnet-4-6":
            self.generate = self.generate_anthropic
        elif model_name == "gemini/gemini-2.5-flash":
            self.generate = self.generate_gemini
        elif model_name == "openai/gpt-5-mini-2025-08-07":
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

class OpenAILLM(LLM):
    _SUPPORTED_MODELS = [
        "gpt-3.5-turbo-0125",
        "gpt-5-mini-2025-08-07",
    ]
    # Which sampling kwargs the client accepts per model
    _MODEL_CHAT_OPTIONS = {
        "gpt-3.5-turbo-0125": frozenset({"top_p", "temperature"}),
        "gpt-5-mini-2025-08-07": frozenset(),
    }

    def __init__(self, model_name: str):
        assert model_name in self._SUPPORTED_MODELS, f"Model {model_name} not supported"
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _chat_completion_create(
        self,
        *,
        messages: List[Dict],
        max_new_tokens: int,
        n: int,
        top_p: float,
        temperature: float,
    ):
        allowed = self._MODEL_CHAT_OPTIONS.get(
            self.model_name,
            frozenset({"top_p", "temperature"}),
        )
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_new_tokens,
            "n": n,
        }
        if "top_p" in allowed:
            kwargs["top_p"] = top_p
        if "temperature" in allowed:
            kwargs["temperature"] = temperature
        return self.client.chat.completions.create(**kwargs)

    def generate(self, 
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
            response = self._chat_completion_create(
                messages=messages,
                max_new_tokens=max_new_tokens,
                n=n_,
                top_p=top_p,
                temperature=temperature,
            )
            responses.extend([(c.message.content or "") for c in response.choices])
        return responses
    
class AnthropicLLM(LLM):
    
    _SUPPORTED_MODELS = [
        "claude-sonnet-4-6",
    ]

    def __init__(self, model_name: str):
        assert model_name in self._SUPPORTED_MODELS, f"Model {model_name} not supported"
        super().__init__(model_name)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    def _split_system_and_messages(messages: List[Dict]) -> Tuple[Optional[str], List[Dict[str, str]]]:
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, turns

    @staticmethod
    def _text_from_message(message: Any) -> str:
        parts = [b.text for b in message.content]
        return "".join(parts)

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 4096,
        n: int = 1,
        top_p: float = 0.96,
        top_k: int = 50,
        temperature: float = 1.0,
    ) -> List[str]:
        system, messages = AnthropicLLM._split_system_and_messages(messages)
        responses: List[str] = []
        for _ in range(n):
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": max_new_tokens,
                "messages": messages,
            }
            if system is not None:
                kwargs["system"] = system
            msg = self.client.messages.create(**kwargs)
            responses.append(self._text_from_message(msg))
        return responses

class GoogleLLM(LLM):
    """Gemini via Google AI Studio / Gemini API (`google-generativeai`)."""

    _SUPPORTED_MODELS = [
        "gemini-2.5-flash",
    ]

    def __init__(self, model_name: str):
        assert model_name in self._SUPPORTED_MODELS, f"Model {model_name} not supported"
        super().__init__(model_name)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    @staticmethod
    def _split_system_and_messages(
        messages: List[Dict],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Extract system text and Gemini-format turns.

        The SDK requires ``{"role": "user"|"model", "parts": [...]}``, not OpenAI's
        ``content`` key — see ``strict_to_content`` / ``content_types.py``.
        """
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            raw = m.get("content", "")
            text = raw if isinstance(raw, str) else str(raw)
            if role == "system":
                system_parts.append(text)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
            elif role == "user":
                contents.append({"role": "user", "parts": [text]})
            else:
                contents.append({"role": "user", "parts": [text]})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, contents

    @staticmethod
    def _text_from_candidate(candidate: Any) -> str:
        content = getattr(candidate, "content", None)
        if content is None:
            return ""
        parts = [p.text for p in content.parts]
        return "".join(parts).strip()

    @staticmethod
    def _texts_from_message(message: Any) -> List[str]:
        """One completion string per requested sample (``candidate_count`` may be > 1)."""
        candidates = getattr(message, "candidates", []) or []
        if not candidates:
            return [(message.text if getattr(message, "text", None) else "").strip()]
        c0 = candidates[0]
        content0 = getattr(c0, "content", None)
        parts0 = list(getattr(content0, "parts", []) or []) if content0 else []
        # Gemini: first candidate may list each alternative as its own Part; other
        # candidates repeat those parts one-at-a-time. Joining parts reproduces the
        # bogus long first ``generation``; take one string per Part on the first row.
        if len(candidates) >= 2 and len(parts0) == len(candidates):
            return [(getattr(p, "text", None) or "").strip() for p in parts0]
        return [GoogleLLM._text_from_candidate(c) for c in candidates]

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 4096,
        n: int = 1,
        top_p: float = 0.96,
        top_k: int = 50,
        temperature: float = 1.0,
    ) -> List[str]:
        system, contents = GoogleLLM._split_system_and_messages(messages)
        if not contents:
            raise ValueError(
                "GoogleLLM needs at least one user or assistant message (besides system)."
            )
        if system is not None:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system,
            )
        else:
            model = genai.GenerativeModel(self.model_name)
            
        responses: List[str] = []
        iterations = (n + 7) // 8

        for i in range(iterations):
            remaining = n - 8 * i
            n_ = min(8, remaining)
            gen_cfg = genai.types.GenerationConfig(
                candidate_count=max(1, n_),
                max_output_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            msg = model.generate_content(
                contents,
                generation_config=gen_cfg,
            )
            responses.extend(GoogleLLM._texts_from_message(msg))
        
        # Align length with requested ``n`` (fewer candidates if blocked/safety).
        if len(responses) < n:
            responses.extend([""] * (n - len(responses)))
        elif len(responses) > n:
            responses = responses[:n]
        return responses

class VllmLLM(LLM):

    _SUPPORTED_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B",
        "openai/gpt-oss-20b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "nvidia/Mistral-NeMo-12B-Instruct",
        "google/gemma-3-12b-it",
    ]
    _THINKING_MODELS = [
        "Qwen/Qwen3-8B",
        "openai/gpt-oss-20b",
    ]
    _THINKING_END_TOKENS = {
        "Qwen/Qwen3-8B": "</think>",
        "openai/gpt-oss-20b": "assistantfinal",
    }
    _MAX_MODEL_LENGTHS = {
        "Qwen/Qwen3-8B": 4096,
        "openai/gpt-oss-20b": 4096,
    }

    def __init__(self, model_name: str):
        assert model_name in self._SUPPORTED_MODELS, f"Model {model_name} not supported"

        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._llm = VLLMEngine(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            max_model_len=self._MAX_MODEL_LENGTHS.get(model_name, 4096),
        )
        self.thinking_end_token = self._THINKING_END_TOKENS.get(model_name, None)

    def _strip_thinking_batch(self, batches: List[List[str]]) -> List[List[str]]:
        end_tok = self.thinking_end_token
        out: List[List[str]] = []
        for group in batches:
            row: List[str] = []
            for s in group:
                if end_tok in s:
                    row.append(s.split(end_tok)[-1].strip())
                else:
                    row.append("")
            out.append(row)
        return out

    def generate_batch(
        self,
        messages_list: Union[List[List[Dict]], List[Dict]],
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        n: int = 1,
        top_p: float = 0.96,
        top_k: int = 0,
        temperature: float = 1.0,
    ) -> List[List[str]]:
        """Batch prompts × ``n`` completions each.

        Outer list aligns with ``messages_list``; inner list has ``n`` strings per prompt.
        """
        if not messages_list:
            return []

        prompts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in messages_list
        ]

        if not do_sample:
            temperature = 0.0
        vllm_top_k = top_k if top_k > 0 else -1

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=vllm_top_k,
        )

        outputs = self._llm.generate(prompts, sampling_params)
        texts: List[List[str]] = [
            [c.text for c in ro.outputs] for ro in outputs
        ]
        if self.model_name in self._THINKING_MODELS:
            texts = self._strip_thinking_batch(texts)
        return texts

class HuggingfaceLLM(LLM):
    
    _SUPPORTED_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B", # 
        "openai/gpt-oss-20b", # 
        "meta-llama/Llama-3.1-8B-Instruct", # pipeline recommended
        "nvidia/Mistral-NeMo-12B-Instruct",
        "google/gemma-3-12b-it"
    ]
    _THINKING_MODELS = [
        "Qwen/Qwen3-8B",
        "openai/gpt-oss-20b"
    ]
    _THINKING_END_TOKENS = {
        "Qwen/Qwen3-8B": "</think>",
        "openai/gpt-oss-20b": "assistantfinal",
    }
    def __init__(self, model_name: str):
        assert model_name in self._SUPPORTED_MODELS, f"Model {model_name} not supported"
        
        super().__init__(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto",
        )
        self.thinking_end_token = self._THINKING_END_TOKENS.get(model_name, None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    
    def generate(self, 
                 messages: List[Dict], 
                 max_new_tokens: int=4096, 
                 do_sample: bool=True,
                 n: int=1,
                 top_p: float=0.96,
                 top_k: int=0,
                 temperature: float=1.0) -> List[str]:
        
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=n,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        outputs = HuggingfaceLLM._parse_outputs(outputs)
        if self.model_name in self._THINKING_MODELS:
            outputs = self._strip_thinking_tokens(outputs)
        return outputs

    @staticmethod
    def _parse_outputs(outputs: List[Any]) -> List[str]:
        return [o['generated_text'][-1]['content'] for o in outputs]

    def _strip_thinking_tokens(self, outputs: List[str]) -> List[str]:
        end_tok = self.thinking_end_token
        out = []
        for o in outputs:
            if end_tok in o:
                out.append(o.split(end_tok)[-1].strip())
            else:
                out.append("")
        return out
    
if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    llm = HuggingfaceLLM(model_name)
    messages = [
        {"role": "user", "content": "This is a natural language inference task. Based on the premise: '%s', create a hypothesis that is entailment or neutral. Output only the hypothesis and nothing else." % ("The weather is rainy and the temperature is 20 degrees Celsius. There is a woman walking on a sidewalk.")}
    ]
    outputs = llm.generate(messages, 
                           n=10)
    print(outputs)