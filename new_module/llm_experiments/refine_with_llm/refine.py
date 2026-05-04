from transformers import AutoModelForCausalLM, AutoTokenizer
from new_module.dev_utils.utils import read_outputs, ravel
import torch
import argparse
import json
from tqdm import tqdm
import time

from new_module.llm_experiments.refine_with_llm.prompt_template_loader import (
    DEFAULT_PROMPT_TEMPLATES_PATH,
    get_edit_prompt_entry,
    load_grammar_refinement_config,
    load_prompt_templates_file,
)

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
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", task=None, consider_prefix=False,
                 num_shots=0, few_shot_path=None, top_p=0.5,
                 prompt_templates_path=None, template_version=None, edit_prompt_key=None):
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
            prompt_templates_path: path to prompt_templates.json (headers, footers, system_message).
                Defaults to prompt_templates.json next to this module.
            template_version: grammar_refinement_versions key (e.g. 1 or 2). None uses
                defaults.grammar_refinement_version from JSON.
            edit_prompt_key: if set, use edit_prompts[key].header plus the shared grammar footer
                for that entry's footer_mode (sentence or continuation). Sets consider_prefix
                from footer_mode (overrides the consider_prefix argument).
        """
        tmpl_path = prompt_templates_path or DEFAULT_PROMPT_TEMPLATES_PATH
        file_data = load_prompt_templates_file(tmpl_path)
        templates = load_grammar_refinement_config(tmpl_path, template_version=template_version)
        self._system_message = templates["system_message"]
        self._sentence_header = templates["sentence_header"]
        self._sentence_footer = templates["sentence_footer"]
        self._continuation_header = templates["continuation_header"]
        self._continuation_footer = templates["continuation_footer"]
        self._examples_section_prefix = templates["examples_section_prefix"]

        self._edit_instruction_header = None
        if edit_prompt_key:
            entry = get_edit_prompt_entry(edit_prompt_key, data=file_data)
            self._edit_instruction_header = entry["header"]
            self.consider_prefix = entry["footer_mode"] == "continuation"
        else:
            self.consider_prefix = consider_prefix

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
        template = (
            self._edit_instruction_header
            if self._edit_instruction_header is not None
            else self._sentence_header
        )

        if num_shots > 0:
            examples = self._load_shots(task, "sentence", num_shots, few_shot_path)
            template += self._examples_section_prefix
            for ex in examples:
                template += f"Original: {ex['original']}\nRefined: {ex['refined']}\n\n"

        template += self._sentence_footer
        return template

    def _build_continuation_template(self, task, num_shots, few_shot_path):
        """Construct the continuation-refinement prompt template with dynamic few-shot examples."""
        template = (
            self._edit_instruction_header
            if self._edit_instruction_header is not None
            else self._continuation_header
        )

        if num_shots > 0:
            examples = self._load_shots(task, "continuation", num_shots, few_shot_path)
            template += self._examples_section_prefix
            for ex in examples:
                template += (f"Prefix: {ex['prefix']}\nOriginal: {ex['original']}\n\n"
                             f"Prefix: {ex['prefix']}\nRefined: {ex['refined']}\n\n")

        template += self._continuation_footer
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
                    {"role": "system", "content": self._system_message},
                    {"role": "user", "content": self.prompt_template % (prefix, input_text, prefix)}
                ]
            else:    
                messages = [
                    {"role": "system", "content": self._system_message},
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
    parser.add_argument(
        '--prompt_templates_path',
        type=str,
        default=None,
        help='Path to prompt_templates.json. Default: prompt_templates.json beside refine.py.',
    )
    parser.add_argument(
        '--template_version',
        type=str,
        default=None,
        help='grammar_refinement_versions key (e.g. 1, 2). Default: JSON defaults.grammar_refinement_version.',
    )
    parser.add_argument(
        '--edit_prompt_key',
        type=str,
        default=None,
        help='Use edit_prompts[key] instruction header + shared grammar footer; sets prefix mode from JSON.',
    )

    args = parser.parse_args()
    
    # define LLM-based text refiner
    refiner = TextRefiner(
        consider_prefix=args.consider_prefix,
        num_shots=args.num_shots,
        few_shot_path=args.few_shot_path,
        task=args.task,
        top_p=args.top_p,
        prompt_templates_path=args.prompt_templates_path,
        template_version=args.template_version,
        edit_prompt_key=args.edit_prompt_key,
    )
    
    # read L&E outputs to refine (=input_data of this script)
    input_data = read_outputs(args.input_path)
    
    # create a deep copy of input_data and call it output_data
    output_data = input_data.copy()
    
    # For each chunk of input_data, refine "text" values in batch and save it into output_data
    refined_texts = []
    
    start_time = time.time()
    # Process in batches
    for i in tqdm(range(0, len(input_data), args.batch_size)):
        chunk = input_data.iloc[i : i + args.batch_size]
        texts_to_refine = chunk['text'].tolist()
        
        if refiner.consider_prefix:
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
    end_time = time.time()
    print(f"Total time taken (seconds): {end_time - start_time}")
    