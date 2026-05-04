## read /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150_postprocessed_edit_candidates_0_99.jsonl
## drop rows with empty hypothesis
## first filter out prompts with "around" or exactly "n" constraints.

## then for each number of constraints, sample equal number of examples.
## save a file. for each number of constraints.

## evaluate nli score.
## also, extract kwargs and then create input_data.json. Also create input_response_data.json. evaluate other constraints.

import pandas as pd
import re

def _constraint_is_exact_count(cid: str, constraint_line: str) -> bool:
    line = constraint_line.strip()
    if cid == "keywords:letter_frequency":
        if not _RE_LETTER_FREQ.search(line):
            return False
        if re.search(r"should appear (at least|less than)\s+\d+", line, re.IGNORECASE):
            return False
        return True
    return False
# Skip only when "around" appears in word-count or all-caps-word-count lines (ifeval_prompts).
_RE_AROUND_WORD_COUNT = re.compile(
    r"Answer with around \d+ words"
    r"(?: The word count does not include the repeated request\.)?",
    re.IGNORECASE,
)

_RE_AROUND_CAPITAL_WORD_COUNT = re.compile(
    r"In your response, words with all capital letters should appear around \d+ times\.",
    re.IGNORECASE,
)


def full_prompt_has_around_count_constraint(full_prompt: str) -> bool:
    return bool(
        _RE_AROUND_WORD_COUNT.search(full_prompt)
        or _RE_AROUND_CAPITAL_WORD_COUNT.search(full_prompt)
    )



data = pd.read_json('/home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150_postprocessed_edit_candidates_0_99.jsonl', lines=True)

data['full_prompt'] 