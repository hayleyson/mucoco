import json, re
from pathlib import Path
import pandas as pd

INPUT_PATH = Path(
    "laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150.jsonl"
)
OUTPUT_PATH = Path(
    "laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150_postprocessed.jsonl"
)

outputs = pd.read_json(INPUT_PATH, lines=True)


def extract_json_content(text):
    try:
        text = text.replace("{{", "{").replace("}}", "}")
        content = json.loads(text)
        return str(content["response"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ""


def remove_prompt(text, prompt):
    if prompt and prompt in text:
        return text.replace(prompt, "")
    return text


def remove_startend_quotation(text):
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def postprocess_generation_text(text, instruction_ids, prompt_text):
    # Order: json parsing -> quotation removal -> prompt removal
    if "detectable_format:json_format" in instruction_ids:
        text = extract_json_content(text)
    if "startend:quotation" in instruction_ids:
        text = remove_startend_quotation(text)
    if "combination:repeat_prompt" in instruction_ids:
        text = remove_prompt(text, prompt_text)
    if not text:
        return "", None, None

    text = text.replace("<hypothesis>...</hypothesis>", "")

    pattern = re.compile(
    r"<hypothesis>(.*?)</hypothesis>", flags=re.IGNORECASE | re.DOTALL
    )
    for m in pattern.finditer(text):
        inner = m.group(1)
        if inner.strip():    
            start, end = m.start(1), m.end(1) 
            return inner.strip(), start, end

    return "", None, None
    # rest = re.sub(r"(?is)<hypothesis>.*?</hypothesis>\s*", "", text)
    # m = re.search(
    #     r"<hypothesis>\s*(.*)", rest, flags=re.IGNORECASE | re.DOTALL
    # )
    # if m and m.group(1).strip():
    #     return m.group(1).strip()
    # # Had `</hypothesis>` somewhere but no non-empty inner content and no recoverable unclosed tail.
    # if closed:
    #     return ""

    # lower = text.lower()
    # if "this is a natural language inference task" in lower:
    #     return ""
    # return text.strip()

def get_prompt_to_be_repeated(prompt_dict):
    
    prompt_text = prompt_dict.get("full_prompt") or ""
    prompt_text = prompt_text.replace("(do not say anything before repeating the request; the request you need to repeat does not include this sentence)", "")
    
    return prompt_text



if __name__ == "__main__":
    parsing_error_count = 0
    for i, (generations, prompt) in enumerate(zip(outputs["generations"], outputs["prompt"])):
        instruction_ids = prompt.get("instruction_id_list") or []
        new_generations = []
        for generation in generations:
            new_generation, start, end = postprocess_generation_text(
                generation["text"], instruction_ids, get_prompt_to_be_repeated(prompt)
            )
            if new_generation == "":
                parsing_error_count += 1
            # else:
            new_generations.append({"text": new_generation, "raw_text": generation["text"], "hypothesis_start": start, "hypothesis_end": end})
        
        outputs.at[i, "generations"] = new_generations
    print(f"Parsing error count: {parsing_error_count}")
    # outputs = outputs.loc[outputs["generations"].apply(lambda x: len(x) > 0)]
    
    outputs.to_json(OUTPUT_PATH, lines=True, orient="records")
