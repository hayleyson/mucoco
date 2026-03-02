from .prompts.utils import load_prompt

class baseLM:
    def __init__(self):
        pass
    
    def set_prompt(self, task: str, prompt_type: str) -> None:
        self.system_prompt, self.user_prompt = load_prompt(task, prompt_type)

    def generate(self, prefix: str):
        pass
