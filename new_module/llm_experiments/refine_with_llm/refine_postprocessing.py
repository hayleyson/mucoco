from transformers import AutoModelForCausalLM, AutoTokenizer
from new_module.dev_utils.utils import read_outputs, ravel
import torch
import argparse
import json
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
    
    CONTINUATION_REFINE_TEMPLATE_HEADER = """Revise the continuation to be grammatically correct and flow naturally from the prefix. Do not modify the prefix, add/remove information, or change the meaning. If no revision is needed, return n/a. Output only the revised continuation or "n/a".
"""
    CONTINUATION_REFINE_TEMPLATE_FOOTER = """### INPUT
Prefix: %s
Original: %s

### OUPUT
Prefix: %s
Refined: 
"""
    SENTENCE_REFINE_TEMPLATE_HEADER = """Correct grammatical errors and improve sentence flow without changing the meaning or content. If no revision is needed, return n/a. Output only the revised sentence or "n/a".
"""
    SENTENCE_REFINE_TEMPLATE_FOOTER = """Original: %s
Refined: 
"""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", task=None, consider_prefix=False,
                 num_shots=0, few_shot_path=None, top_p=0.5):
        """
        Args:
            model_name: name of the LLM to refine the given texts.
            consider_prefix: indicates whether flow from the prefix must be considered
                when revising the sentence.
            num_shots: number of few-shot examples to include in the prompt.
                0 means no examples.
            few_shot_path: path to a JSON file containing few-shot examples.
                Required when num_shots > 0.
                - Sentence mode: list of {"original": ..., "refined": ...} objects.
                - Continuation mode: list of {"prefix": ..., "original": ..., "refined": ...} objects.
                "refined" may be "n/a" in both cases.
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
        
        # Configure tokenizer for batch processing
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.task = task
        self.consider_prefix = consider_prefix
        self.top_p = top_p

        print(f"task: {task}")

        if self.consider_prefix:
            self.prompt_template = self._build_continuation_template(task, num_shots, few_shot_path)
        else:
            self.prompt_template = self._build_sentence_template(task, num_shots, few_shot_path)

    def _load_shots(self, task, template_type, num_shots, few_shot_path):
        """Load and truncate few-shot examples from a JSON file."""
        if few_shot_path is None:
            raise ValueError("few_shot_path must be provided when num_shots > 0.")
        with open(few_shot_path, 'r') as f:
            examples = json.load(f)
        examples = examples[template_type][task]
        if len(examples) < num_shots:
            print(f"Warning: requested {num_shots} shots but only {len(examples)} "
                  f"examples available. Using {len(examples)}.")
            num_shots = len(examples)
        return examples[:num_shots]

    def _build_sentence_template(self, task, num_shots, few_shot_path):
        """Construct the sentence-refinement prompt template with dynamic few-shot examples."""
        template = self.SENTENCE_REFINE_TEMPLATE_HEADER

        if num_shots > 0:
            examples = self._load_shots(task, "sentence", num_shots, few_shot_path)
            template += "\n### EXAMPLES\n"
            for ex in examples:
                template += f"Original: {ex['original']}\nRefined: {ex['refined']}\n\n"

        template += self.SENTENCE_REFINE_TEMPLATE_FOOTER
        return template

    def _build_continuation_template(self, task, num_shots, few_shot_path):
        """Construct the continuation-refinement prompt template with dynamic few-shot examples."""
        template = self.CONTINUATION_REFINE_TEMPLATE_HEADER

        if num_shots > 0:
            examples = self._load_shots(task, "continuation", num_shots, few_shot_path)
            template += "\n### EXAMPLES\n"
            for ex in examples:
                template += (f"Prefix: {ex['prefix']}\nOriginal: {ex['original']}\n\n"
                             f"Prefix: {ex['prefix']}\nRefined: {ex['refined']}\n\n")

        template += self.CONTINUATION_REFINE_TEMPLATE_FOOTER
        return template

    def refine_batch(self, input_texts, prefixes=None):
        """Refines a batch of input texts.
        
        Args:
            input_texts (list[str]): List of texts to refine.
            prefixes (list[str], optional): List of prefixes for continuation mode.
        """
        if prefixes is None:
            prefixes = [""] * len(input_texts)
            
        all_messages = []
        for input_text, prefix in zip(input_texts, prefixes):
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
            all_messages.append(messages)
        
        texts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in all_messages
        ]
        
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample=True,
            top_p=self.top_p,
            max_new_tokens=512
        )
        
        # Extract only the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        refined_results = []
        for response, original_text in zip(responses, input_texts):
            if ("n/a" in response.lower()):
                refined_results.append(original_text)
            else:
                refined_results.append(response)
        
        return refined_results



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--task', type=str)
    parser.add_argument('--consider_prefix', action='store_true', default=False)
    parser.add_argument('--num_shots', type=int, default=0,
                        help='Number of few-shot examples to include in the prompt.')
    parser.add_argument('--few_shot_path', type=str, default=None,
                        help='Path to a JSON file containing few-shot examples '
                             '[{"original": ..., "refined": ...}, ...]. '
                             'Required when --num_shots > 0 and --consider_prefix is not set.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for LLM inference.')
    parser.add_argument('--top_p', type=float, default=0.5,
                        help='Top-p (nucleus) sampling parameter.')

    args = parser.parse_args()
    
    # define LLM-based text refiner
    refiner = TextRefiner(
        consider_prefix=args.consider_prefix,
        num_shots=args.num_shots,
        few_shot_path=args.few_shot_path,
        task=args.task,
        top_p=args.top_p
    )
    
    # read L&E outputs to refine (=input_data of this script)
    input_data = read_outputs(args.input_path)
    
    # create a deep copy of input_data and call it output_data
    output_data = input_data.copy()
    
    # For each chunk of input_data, refine "text" values in batch and save it into output_data
    refined_texts = []
    
    # Process in batches
    for i in tqdm(range(0, len(input_data), args.batch_size)):
        chunk = input_data.iloc[i : i + args.batch_size]
        texts_to_refine = chunk['text'].tolist()
        
        if args.consider_prefix:
            prefixes = chunk['prompt'].tolist()
        else:
            prefixes = None
            
        batch_refined = refiner.refine_batch(texts_to_refine, prefixes=prefixes)
        refined_texts.extend(batch_refined)
        
    # format change to match L&E outputs
    output_data['text'] = refined_texts
    output_data = ravel(output_data)
    # save the results
    output_data.to_json(args.output_path, lines=True, orient="records")
    