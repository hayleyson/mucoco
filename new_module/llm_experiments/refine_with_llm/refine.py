from transformers import AutoModelForCausalLM, AutoTokenizer
from new_module.dev_utils.utils import read_outputs, ravel
import argparse
import json
from tqdm import tqdm
import time
import pandas as pd

from new_module.evaluation.evaluate_pipeline import run_generation_evaluation
from new_module.llm_experiments.refine_with_llm.prompt_template_loader import (
    DEFAULT_PROMPT_TEMPLATES_PATH,
    load_grammar_refinement_prompt_templates,
    load_few_shot_examples
)

_DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

# Prefix / continuation mode is fixed per task (not a CLI flag).
_TASK_CONSIDER_PREFIX = {
    "toxicity": True,
    "nli": False,
    "set_lconvqa": False,
    "nli_toxicity": False,
}
class TextRefiner:
    """Refines the grammar and fluency of input text using an LLM."""

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        task=None,
        num_shots=0,
        few_shot_path=None,
        top_p=0.5,
        prompt_templates_path=None,
        template_version=None,
        use_vllm=False,
    ):
        tmpl_path = prompt_templates_path or DEFAULT_PROMPT_TEMPLATES_PATH
        templates = load_grammar_refinement_prompt_templates(tmpl_path, template_version=template_version)
        self._system_message = templates.get("system_message") or _DEFAULT_SYSTEM_MESSAGE
        self._sentence_header = templates["sentence_header"]
        self._sentence_footer = templates["sentence_footer"]
        self._continuation_header = templates["continuation_header"]
        self._continuation_footer = templates["continuation_footer"]
        self._examples_section_prefix = templates["examples_section_prefix"]

        self.consider_prefix = _TASK_CONSIDER_PREFIX[task]

        self.use_vllm = use_vllm
        self.top_p = top_p
        self.task = task

        print(f"Loading tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_vllm:
            from vllm import LLM as VLLMEngine

            print(f"Loading vLLM engine: {model_name}...")
            self._vllm = VLLMEngine(model=model_name, gpu_memory_utilization=0.9)
            self.model = None
        else:
            print(f"Loading Hugging Face model: {model_name}...")
            self._vllm = None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto",
            )

        print(f"task: {task}")

        if self.consider_prefix:
            self.prompt_template = self._build_continuation_template(task, num_shots, few_shot_path)
        else:
            self.prompt_template = self._build_sentence_template(task, num_shots, few_shot_path)

    def _build_sentence_template(self, task, num_shots, few_shot_path):
        template = self._sentence_header

        if num_shots > 0:
            examples = load_few_shot_examples(few_shot_path, task, num_shots)
            template += self._examples_section_prefix
            for ex in examples:
                template += f"Original: {ex['original']}\nRefined: {ex['refined']}\n\n"

        template += self._sentence_footer
        return template

    def _build_continuation_template(self, task, num_shots, few_shot_path):
        template = self._continuation_header

        if num_shots > 0:
            examples = load_few_shot_examples(few_shot_path, task, num_shots)
            template += self._examples_section_prefix
            for ex in examples:
                template += (
                    f"Prefix: {ex['prefix']}\nOriginal: {ex['original']}\n\n"
                    f"Prefix: {ex['prefix']}\nRefined: {ex['refined']}\n\n"
                )

        template += self._continuation_footer
        return template

    def _prompt_strings(self, input_texts: list[str], prefixes: list[str] | None = None) -> list[str]:
        if prefixes is None:
            prefixes = [""] * len(input_texts)
        if len(prefixes) != len(input_texts):
            raise ValueError("prefixes must be the same length as input_texts.")

        all_messages = []
        for input_text, prefix in zip(input_texts, prefixes):
            if self.consider_prefix:
                messages = [
                    {"role": "system", "content": self._system_message},
                    {"role": "user", "content": self.prompt_template % (prefix, input_text, prefix)},
                ]
            else:
                messages = [
                    {"role": "system", "content": self._system_message},
                    {"role": "user", "content": self.prompt_template % input_text},
                ]
            all_messages.append(messages)

        return [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]

    def refine_batch(self, input_texts: list[str], prefixes: list[str] | None = None) -> list[str]:
        """Hugging Face batched generation (bounded by ``batch_size`` in the driver loop)."""
        if self.use_vllm:
            raise RuntimeError("refine_batch is for Hugging Face only; use refine_batch_vllm when use_vllm=True.")

        texts = self._prompt_strings(input_texts, prefixes)
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample=True,
            top_p=self.top_p,
            max_new_tokens=512,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return self._responses_to_refined(input_texts, responses)

    def refine_batch_vllm(self, input_texts: list[str], prefixes: list[str] | None = None) -> list[str]:
        """Single vLLM call over all prompts (vLLM schedules batching internally)."""
        if not self.use_vllm:
            raise RuntimeError("refine_batch_vllm requires use_vllm=True.")

        from vllm import SamplingParams

        texts = self._prompt_strings(input_texts, prefixes)
        sampling_params = SamplingParams(max_tokens=512, top_p=self.top_p, temperature=1.0)
        outputs = self._vllm.generate(texts, sampling_params)
        responses = [o.outputs[0].text for o in outputs]
        return self._responses_to_refined(input_texts, responses)

    @staticmethod
    def _responses_to_refined(input_texts: list[str], responses: list[str]) -> list[str]:
        refined_results = []
        for response, original_text in zip(responses, input_texts):
            if "n/a" in response.lower():
                refined_results.append(original_text)
            else:
                refined_results.append(response)
        return refined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task id: toxicity (prefix/continuation), nli or set_lconvqa (sentence-only).",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--num_shots",
        type=int,
        default=0,
        help="Number of few-shot examples to include in the prompt.",
    )
    parser.add_argument(
        "--few_shot_path",
        type=str,
        default=None,
        help="Path to a JSON file containing few-shot examples. Required when --num_shots > 0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for Hugging Face inference (ignored with --use_vllm).",
    )
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument(
        "--prompt_templates_path",
        type=str,
        default=None,
        help="Path to prompt_templates.json. Default: prompt_templates.json beside refine.py.",
    )
    parser.add_argument(
        "--template_version",
        type=str,
        default=None,
        required=True,
        help="grammar_refinement_versions key (e.g. 1, 2). Default: JSON defaults or 2.",
    )
    parser.add_argument(
        "--original_text_path",
        type=str,
        default=None,
        help="Optional JSONL with a text/generation column; replaces input `text` column when lengths match.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Run inference with vLLM instead of Hugging Face generate().",
    )
    args = parser.parse_args()

    refiner = TextRefiner(
        model_name=args.model_name,
        num_shots=args.num_shots,
        few_shot_path=args.few_shot_path,
        task=args.task,
        top_p=args.top_p,
        prompt_templates_path=args.prompt_templates_path,
        template_version=args.template_version,
        use_vllm=args.use_vllm,
    )

    input_data = read_outputs(args.input_path)
    output_data = input_data.copy()
    refined_texts = []

    start_time = time.time()
    if args.use_vllm:
        texts_to_refine = input_data["text"].tolist()
        prefixes = input_data["prompt"].tolist() if refiner.consider_prefix else None
        refined_texts = refiner.refine_batch_vllm(texts_to_refine, prefixes=prefixes)
    else:
        refined_texts = []
        for i in tqdm(range(0, len(input_data), args.batch_size)):
            chunk = input_data.iloc[i : i + args.batch_size]
            texts_to_refine = chunk["text"].tolist()
            if refiner.consider_prefix:
                prefixes = chunk["prompt"].tolist()
            else:
                prefixes = None
            refined_texts.extend(refiner.refine_batch(texts_to_refine, prefixes=prefixes))

    output_data["text"] = refined_texts
    output_data = ravel(output_data)
    output_data.to_json(args.output_path, lines=True, orient="records")
    
    end_time = time.time()
    with open(args.output_path.replace('.jsonl', '.time'), 'w') as f:
        f.write(str(end_time - start_time) + '\n')


    if args.task == "toxicity":
        run_generation_evaluation(
            "",
            args.output_path,
            "toxicity,toxicity-int,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
            source_file_path=args.original_text_path,
            task=args.task,
        )  # 시간 문제로, perspective api 제외
    elif args.task == "nli":
        run_generation_evaluation(
            "",
            args.output_path,
            "nli,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
            source_file_path=args.original_text_path,
            task=args.task,
        )  
    elif args.task == "set_lconvqa":
        run_generation_evaluation(
        "",
        args.output_path,
        "set-consistency,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
        source_file_path=args.original_text_path,
        task=args.task,
    )  
    elif args.task == "nli_toxicity":
        run_generation_evaluation(
        "",
        args.output_path,
        "toxicity,nli,ppl-qwen,dist-n,repetition,fluency,contents-preservation",
        source_file_path=args.original_text_path,
        task=args.task, 
    )