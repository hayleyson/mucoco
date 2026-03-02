import os
from openai import OpenAI
from .base import baseLM
from .prompts.utils import load_prompt
from typing import List

class OpenAIChatLM(baseLM):
    """Base language model class that calls OpenAI Chat API."""
    
    def __init__(self, model_name: str, task: str, prompt_type: str):
        """
        Initialize the base language model.
        
        Args:
            model: Name of the OpenAI model (e.g., 'gpt-3.5-turbo').
        """
        self.api_key = os.getenv("OPENAI_API_KEY", None)
        
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables.")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        self.set_prompt(task, prompt_type)
        
    def generate(self, prefix: str, n: int, max_new_tokens: int, top_p: float = 0.96) -> List[str]:
        """
        Generate completions for the provided prefix.
        
        Args:
            prefix: Text to continue.
            n: Number of completions.
            max_new_tokens: Max tokens for generation.
            top_p: Nucleus sampling threshold.
        """
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            # If the prompt includes {prefix}, inject it. Otherwise use the prompt as is
            {"role": "user", "content": self.user_prompt.format(prefix=prefix) if "{prefix}" in self.user_prompt else self.user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=n,
            messages=messages
        )
        
        generations = [x.message.content for x in response.choices]
        
        return generations