from transformers import AutoModelForCausalLM, AutoTokenizer
from new_module.dev_utils.utils import read_outputs, ravel
import torch
import argparse
from tqdm import tqdm


class TextRefiner:
    
    """Refines the grammar and fluency of input text using an LLM.

    Attributes:
        model (transformers.AutoModelForCausalLM): The language model used for refinement.
        tokenizer (transformers.AutoTokenizer): The tokenizer associated with the model.
        prompt_template (str): A reusable prompt template containing instructions and few-shot examples.

    Example:
        >>> refiner = TextRefiner()
        >>> result = refiner.refine("He go to school yesterday.")
    """
    
    CONTINUATION_REFINE_TEMPLATE = """### INSTRUCTIONS    
You are an expert editor. Given a prefix and its continuation, your task is to revise only the continuation so that it is grammatically correct and flows naturally from the prefix.
- Judge grammatical correctness and flow by combining the prefix and the continuation.
- Ensure the revised continuation flows naturally from the prefix.
- Avoid modifying the prefix.
- Avoid adding new information or removing existing information. 
- Avoid changing the intent, tone, or facts. 
- Only revise wording, grammar, or phrasing for clarity and naturalness. 
- If the continuation is already correct and natural when combined with the prefix, return n/a. 
- Only output the revised continuation or "n/a". 

### INPUT
Prefix: %s
Original: %s

### OUPUT
Prefix: %s
Refined: 
"""
    SENTENCE_REFINE_TEMPLATE = """### INSTRUCTIONS    
You are an expert editor. Your task is to correct grammatical errors and improve sentence flow while preserving the original meaning and content exactly. 
- Avoid adding new information or removing existing information. 
- Avoid changing the intent, tone, or facts. 
- Only revise wording, grammar, or phrasing for clarity and naturalness. 
- If the sentence is already correct and natural, return n/a. 
- Only output the revised sentence or "n/a". 

### EXAMPLES
Original: She don't like going to the office on Mondays. 
Refined: She doesn't like going to the office on Mondays. 

Original: We discussed the new strategy during the the meeting. 
Refined: We discussed the new strategy during the meeting. 

Original: A man is standing next to a chair. 
Refined: n/a 

Original: %s
Refined: 
"""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", consider_prefix = False):
        """
        Args
        model_name: name of the LLM to refine the given texts.
        consider_prefix: indicates whether flow from the prefix must be considered when revising the sentence.
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
        self.consider_prefix = consider_prefix
        
        if self.consider_prefix:
            self.prompt_template = self.CONTINUATION_REFINE_TEMPLATE
        else:
            self.prompt_template = self.SENTENCE_REFINE_TEMPLATE

    def refine(self, input_text, prefix= ""):
        if self.consider_prefix:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": self.prompt_template % (prefix, input_text, prefix)}
            ]
        else:    
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": self.prompt_template % input_text}
            ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample = True,
            top_p = 0.5,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if ("n/a" in response) or ("N/A" in response):
            return input_text
        
        return response



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--consider_prefix', type=bool, default=False)
    
    args = parser.parse_args()
    
    # define LLM-based text refiner
    refiner = TextRefiner(consider_prefix=args.consider_prefix)
    
    # read L&E outputs to refine (=input_data of this script)
    input_data = read_outputs(args.input_path)
    
    # create a deep copy of input_data and call it output_data
    output_data = input_data.copy()
    
    # for each row of input_data, refine "text" value and save it into output_data
    refined_texts = []
    for i, row in tqdm(input_data.iterrows()):
        
        refined_texts.append(refiner.refine(row['text'], prefix=row['prompt'] if args.consider_prefix else ""))
        
    # format change to match L&E outputs
    output_data['text'] = refined_texts
    output_data = ravel(output_data)
    # save the results
    output_data.to_json(args.output_path, lines=True, orient="records")
    