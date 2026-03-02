import os
import yaml

def load_prompt(task, prompt_type, args=None):
    """
    Loads system and user prompts from YAML files based on the task and prompt_type.
    """

    base_dir = os.path.dirname(__file__)
    
    if "plain" in prompt_type:
        file_path = "continuation.yaml"
    else:
        file_path = f"{task}/{prompt_type}.yaml"
    
    full_path = os.path.join(base_dir, file_path)
    
    try:
        with open(full_path, 'r') as f:
            data = yaml.safe_load(f)
            
        system_prompt = data.get("system", "")
        user_prompt = data.get("user", "")
    except (FileNotFoundError, yaml.YAMLError):
        raise FileNotFoundError(f"Prompt file not found: {full_path}")
    
    return (system_prompt, user_prompt)
