#!/usr/bin/env python3
"""
Convert *_postprocessed.jsonl (schema from gpt_api_generate nli_ifeval) into
(1) input_data-style records and (2) ``input_response_data.jsonl``-style records,
with one line per sampled generation (aligned line order in both files):

  Primary (-o): {"key", "prompt", "instruction_id_list", "kwargs"}
  Companion:    {"prompt", "response"}  (same ``prompt`` as primary; ``response`` from generations)

The same ``prompt`` / ``instruction_id_list`` / ``kwargs`` are repeated on each
primary line for every ``record["generations"]`` entry; companion lines hold the
matching ``response`` text. If ``generations`` is empty, one pair of lines is written
and ``response`` is ``""``.

Default ``--response-output`` is ``input_response_data.jsonl`` next to ``-o`` when ``-o``
is set, else ``./input_response_data.jsonl`` (override explicitly).

Provide **at least one** of ``--input-jsonl`` or ``--input-response-jsonl``:

- ``--input-jsonl`` only: read postprocessed JSONL and write **input_data** lines to ``-o``
  (one line per generation; ``--key-start`` applies). Does not read a response file.

- ``--input-response-jsonl`` only: read a response JSONL/text file and write
  **input_response_data** lines to ``--response-output``. Each line: if JSON object has
  ``prompt`` and ``response``, optional around-count skip uses ``prompt``; otherwise lines
  are parsed like external model outputs (``text`` / ``response`` field, or ``--response-format text``)
  and wrapped as ``{"prompt": "", "response": ...}``.

- **Both**: same as before — postprocessed rows from ``--input-jsonl`` drive prompt metadata;
  response text comes from ``--input-response-jsonl``. Skipped postprocessed rows (around-count)
  do **not** consume response lines; pairing uses ``--response-count``:

  - ``auto`` (default): per kept row, read ``max(1, len(record["generations"]))`` lines
    from the response file (use empty ``{}`` placeholders in the prompt JSONL if you only
    need the count).
  - ``one``: exactly one response line per kept prompt row.

Each response line (paired mode) is either raw text (``--response-format text``) or JSON (default) with
a string in ``text`` or ``response``.

- prompt is taken from prompt.full_prompt (full user message, including NLI preamble
  and IFEval-style constraints), not premise-only prompt.text.

- kwargs are recovered from the per-constraint lines in full_prompt. Generation uses
  new_module.data.nli_ifeval.ifeval_prompts: constraint lines are emitted in the same
  order as instruction_id_list, then the fixed hypothesis-tag line is appended.

- CLI conversion skips rows only when full_prompt matches an "around" **count** constraint
  from ifeval_prompts: word count ("Answer with around N words") or all-caps word count
  ("… all capital letters should appear around N times."). Incidental "around" in the
  premise is kept.

Kwargs keys mirror google-research instruction_following_eval / input_data.jsonl:

  length_constraints:number_words
      {"relation": "at least" | "less than", "num_words": int}
  keywords:letter_frequency
      {"letter": str, "let_frequency": int, "let_relation": "at least" | "less than"}
  change_case:capital_word_frequency
      {"capital_frequency": int, "capital_relation": "at least" | "less than"}
  combination:repeat_prompt
      {"prompt_to_repeat": str}  (full NLI base request; see recover_prompt_to_repeat)
  detectable_format:json_format, change_case:*, punctuation:no_comma,
  startend:quotation
      {}

Template → kwargs mapping (ifeval_prompts.py vs official IFEval NumberOfWords / checkers):

  - "Answer with at least N words" → relation "at least", num_words N
  - "Answer with at most N words" → relation "less than", num_words N+1  (strict < N+1)
  - Rows with "Answer with around N words" or all-caps "around N times" are skipped in main().

  - Letter line has no relation in our template ("should appear N times"):
      → treated as an **exact** count: two kwargs entries (duplicate instruction id)
        {"let_relation": "at least", "let_frequency": N} and
        {"let_relation": "less than", "let_frequency": N+1} (same letter), so the
        combined checks enforce exactly N letter occurrences.

  - "words with all capital letters should appear at least|at most N times":
      at least → capital_relation "at least", capital_frequency N
      at most  → capital_relation "less than", capital_frequency N+1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Literal, TextIO

# Fixed tail from ifeval_prompts.FORMAT_INSTRUCTION (single line in generated prompts).
_HYPOTHESIS_TAIL_PREFIX = "Enclose the hypothesis in <hypothesis>"

# Regexes aligned with ifeval_prompts.constraint_specs (after .format(...)).
_RE_NUMBER_WORDS = re.compile(
    r"Answer with (at least|at most) (\d+) words"
    r"(?: The word count does not include the repeated request\.)?",
    re.IGNORECASE,
)
_RE_LETTER_FREQ = re.compile(
    r"In your response, the letter ([a-zA-Z]) should appear (\d+) times\.",
    re.IGNORECASE,
)
_RE_CAPITAL_WORD_FREQ = re.compile(
    r"In your response, words with all capital letters should appear "
    r"(at least|at most) (\d+) times\.",
    re.IGNORECASE,
)

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


def _split_base_and_constraint_lines(
    full_prompt: str, instruction_id_list: list[str]
) -> tuple[str, list[str]]:
    """Split full_prompt into base text and one line per instruction_id (in order)."""
    lines = full_prompt.split("\n")
    n_ids = len(instruction_id_list)
    if not lines:
        raise ValueError("empty full_prompt")
    last = lines[-1].strip()
    if not last.startswith(_HYPOTHESIS_TAIL_PREFIX):
        raise ValueError(
            f"Expected last line to start with {_HYPOTHESIS_TAIL_PREFIX!r}, got {last[:80]!r}..."
        )
    if len(lines) < n_ids + 2:
        raise ValueError(
            f"full_prompt has {len(lines)} lines, need at least {n_ids + 2} "
            f"(base + {n_ids} constraints + format line)"
        )
    base_lines_count = len(lines) - n_ids - 1
    base_text = "\n".join(lines[:base_lines_count])
    constraint_lines = lines[base_lines_count : base_lines_count + n_ids]
    return base_text, constraint_lines


def recover_prompt_to_repeat(base_text: str) -> str:
    """NLI 'request' to repeat: entire premise task block (before sampled constraints)."""
    return base_text.strip()


def _kwargs_number_words(constraint_line: str) -> dict:
    m = _RE_NUMBER_WORDS.search(constraint_line.strip())
    if not m:
        raise ValueError(f"Could not parse length_constraints line: {constraint_line!r}")
    cond, n_s = m.group(1).lower(), m.group(2)
    n = int(n_s)
    if cond == "at least":
        return {"relation": "at least", "num_words": n}
    if cond == "at most":
        return {"relation": "less than", "num_words": n + 1}
    raise ValueError(f"Unexpected word-count condition {cond!r}")


def _kwargs_letter_frequency(constraint_line: str) -> dict:
    m = _RE_LETTER_FREQ.search(constraint_line.strip())
    if not m:
        raise ValueError(f"Could not parse keywords:letter_frequency line: {constraint_line!r}")
    letter, n_s = m.group(1).lower(), m.group(2)
    return {
        "letter": letter,
        "let_frequency": int(n_s),
        "let_relation": "at least",
    }


def _kwargs_capital_word_frequency(constraint_line: str) -> dict:
    m = _RE_CAPITAL_WORD_FREQ.search(constraint_line.strip())
    if not m:
        raise ValueError(
            f"Could not parse change_case:capital_word_frequency line: {constraint_line!r}"
        )
    cond, n_s = m.group(1).lower(), m.group(2)
    n = int(n_s)
    if cond == "at least":
        return {"capital_relation": "at least", "capital_frequency": n}
    if cond == "at most":
        return {"capital_relation": "less than", "capital_frequency": n + 1}
    raise ValueError(f"Unexpected capital-word condition {cond!r}")


def _constraint_is_exact_count(cid: str, constraint_line: str) -> bool:
    line = constraint_line.strip()
    if cid == "keywords:letter_frequency":
        if not _RE_LETTER_FREQ.search(line):
            return False
        if re.search(r"should appear (at least|less than)\s+\d+", line, re.IGNORECASE):
            return False
        return True
    return False


def _exact_count_pair_kwargs(cid: str, constraint_line: str) -> tuple[dict, dict]:
    line = constraint_line.strip()
    if cid == "keywords:letter_frequency":
        m = _RE_LETTER_FREQ.search(line)
        if not m:
            raise ValueError(f"letter exact pair: {constraint_line!r}")
        letter, n = m.group(1).lower(), int(m.group(2))
        lo = {"letter": letter, "let_relation": "at least", "let_frequency": n}
        hi = {"letter": letter, "let_relation": "less than", "let_frequency": n + 1}
        return lo, hi
    raise ValueError(f"exact pair not defined for {cid!r}")


def expand_exact_count_instruction_pairs(
    instruction_id_list: list[str],
    kwargs: list[dict],
    constraint_lines: list[str],
) -> tuple[list[str], list[dict]]:
    """Duplicate instruction ids + kwargs where the prompt fixes an exact count."""
    if len(instruction_id_list) != len(kwargs) or len(instruction_id_list) != len(
        constraint_lines
    ):
        raise ValueError("Mismatched lengths for instruction_id_list / kwargs / lines")
    new_ids: list[str] = []
    new_kwargs: list[dict] = []
    for cid, kw, line in zip(instruction_id_list, kwargs, constraint_lines):
        if _constraint_is_exact_count(cid, line):
            lo, hi = _exact_count_pair_kwargs(cid, line)
            new_ids.extend([cid, cid])
            new_kwargs.extend([lo, hi])
        else:
            new_ids.append(cid)
            new_kwargs.append(kw)
    return new_ids, new_kwargs


def recover_kwargs(full_prompt: str, instruction_id_list: list[str]) -> list[dict]:
    base_text, constraint_lines = _split_base_and_constraint_lines(
        full_prompt, instruction_id_list
    )
    return recover_kwargs_for_lines(base_text, instruction_id_list, constraint_lines)


def recover_kwargs_for_lines(
    base_text: str,
    instruction_id_list: list[str],
    constraint_lines: list[str],
) -> list[dict]:
    if len(constraint_lines) != len(instruction_id_list):
        raise ValueError(
            f"constraint line count {len(constraint_lines)} != "
            f"len(instruction_id_list) {len(instruction_id_list)}"
        )
    out: list[dict] = []
    for cid, line in zip(instruction_id_list, constraint_lines):
        if cid == "length_constraints:number_words":
            out.append(_kwargs_number_words(line))
        elif cid == "keywords:letter_frequency":
            out.append(_kwargs_letter_frequency(line))
        elif cid == "change_case:capital_word_frequency":
            out.append(_kwargs_capital_word_frequency(line))
        elif cid == "combination:repeat_prompt":
            out.append({"prompt_to_repeat": recover_prompt_to_repeat(base_text)})
        elif cid in (
            "detectable_format:json_format",
            "change_case:english_capital",
            "change_case:english_lowercase",
            "punctuation:no_comma",
            "startend:quotation",
        ):
            out.append({})
        else:
            raise ValueError(f"Unknown instruction_id for nli_ifeval: {cid!r}")
    return out


def _next_nonempty_line(fin: TextIO) -> str | None:
    while True:
        line = fin.readline()
        if line == "":
            return None
        s = line.strip()
        if s:
            return s


def parse_response_line(line: str, fmt: Literal["json", "text"]) -> str:
    if fmt == "text":
        return line
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError(f"Response JSON line must be an object, got {type(obj).__name__}")
    if "text" in obj:
        v = obj["text"]
    elif "response" in obj:
        v = obj["response"]
    else:
        raise ValueError("Response JSON object needs a 'text' or 'response' string field")
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ValueError(f"Response field must be str, got {type(v).__name__}")
    return v


def read_n_response_texts(
    fin: TextIO,
    n: int,
    fmt: Literal["json", "text"],
) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        raw = _next_nonempty_line(fin)
        if raw is None:
            raise ValueError(
                f"Unexpected end of response file: needed {n} lines, got {len(out)}"
            )
        out.append(parse_response_line(raw, fmt))
    return out


def row_from_postprocessed(record: dict, key: int) -> dict:
    prompt_block = record["prompt"]
    instruction_id_list = list(prompt_block.get("instruction_id_list") or [])
    full_prompt = prompt_block.get("full_prompt")
    if not full_prompt:
        raise ValueError("Missing prompt.full_prompt; cannot build input_data row.")
    base_text, constraint_lines = _split_base_and_constraint_lines(
        full_prompt, instruction_id_list
    )
    kwargs = recover_kwargs_for_lines(base_text, instruction_id_list, constraint_lines)
    instruction_id_list, kwargs = expand_exact_count_instruction_pairs(
        instruction_id_list, kwargs, constraint_lines
    )
    return {
        "key": key,
        "prompt": full_prompt,
        "instruction_id_list": instruction_id_list,
        "kwargs": kwargs,
    }


def input_and_response_rows_for_prompt(
    record: dict,
    key_start: int,
    response_texts: list[str],
) -> tuple[list[dict], list[dict]]:
    """Build aligned rows using ``record`` for prompt metadata and ``response_texts`` for answers."""
    payload = row_from_postprocessed(record, 0)
    base_prompt = payload["prompt"]
    base_ids = payload["instruction_id_list"]
    base_kwargs = payload["kwargs"]
    input_rows: list[dict] = []
    response_rows: list[dict] = []

    def append_pair(key: int, response_text: str) -> None:
        input_rows.append(
            {
                "key": key,
                "prompt": base_prompt,
                "instruction_id_list": base_ids,
                "kwargs": base_kwargs,
            }
        )
        response_rows.append({"prompt": base_prompt, "response": response_text})

    if not response_texts:
        append_pair(key_start, "")
    else:
        for offset, text in enumerate(response_texts):
            append_pair(key_start + offset, text)
    return input_rows, response_rows


def input_and_response_rows_from_postprocessed(
    record: dict, key_start: int
) -> tuple[list[dict], list[dict]]:
    """Aligned input_data rows and for_ifeval input_response_data rows (prompt + response)."""
    generations = record.get("generations") or []
    if not generations:
        texts: list[str] = []
    else:
        texts = [
            gen.get("text", "") if isinstance(gen, dict) else "" for gen in generations
        ]
    return input_and_response_rows_for_prompt(record, key_start, texts)


def _default_response_output(input_data_output: Path | None) -> Path:
    if input_data_output is not None:
        return input_data_output.with_name("input_response_data.jsonl")
    return Path("input_response_data.jsonl")


def process_input_data_only(
    fin: TextIO,
    fout: TextIO,
    key_start: int,
) -> None:
    """Postprocessed JSONL → input_data JSONL only."""
    n = 0
    for line in fin:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        full_prompt = (record.get("prompt") or {}).get("full_prompt") or ""
        if full_prompt_has_around_count_constraint(full_prompt):
            continue
        in_rows, _ = input_and_response_rows_from_postprocessed(record, key_start + n)
        for out in in_rows:
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        n += len(in_rows)


def process_response_data_only(
    fin: TextIO,
    fout_resp: TextIO,
    response_format: Literal["json", "text"],
) -> None:
    """Normalize or filter lines into input_response_data JSONL (standalone)."""
    for raw in fin:
        raw = raw.strip()
        if not raw:
            continue
        if response_format == "text":
            out: dict = {"prompt": "", "response": raw}
        else:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and "prompt" in obj and "response" in obj:
                prompt = obj.get("prompt") or ""
                if full_prompt_has_around_count_constraint(prompt):
                    continue
                out = obj
            else:
                text = parse_response_line(raw, "json")
                out = {"prompt": "", "response": text}
        fout_resp.write(json.dumps(out, ensure_ascii=False) + "\n")


def process_paired_postprocessed_and_responses(
    fin: TextIO,
    fin_resp: TextIO,
    fout: TextIO,
    fout_resp: TextIO,
    key_start: int,
    response_count: Literal["auto", "one"],
    response_format: Literal["json", "text"],
) -> None:
    """Postprocessed + external response file → both outputs (aligned)."""
    n = 0
    for line in fin:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        full_prompt = (record.get("prompt") or {}).get("full_prompt") or ""
        if full_prompt_has_around_count_constraint(full_prompt):
            continue
        if response_count == "one":
            n_resp = 1
        else:
            gens = record.get("generations") or []
            n_resp = max(1, len(gens))
        texts = read_n_response_texts(fin_resp, n_resp, response_format)
        in_rows, resp_rows = input_and_response_rows_for_prompt(record, key_start + n, texts)
        for out, resp in zip(in_rows, resp_rows):
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout_resp.write(json.dumps(resp, ensure_ascii=False) + "\n")
        n += len(in_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Postprocessed JSONL (gpt_api_generate nli_ifeval). Requires -o/--output.",
    )
    parser.add_argument(
        "--input-response-jsonl",
        type=Path,
        default=None,
        help="Response JSONL/text: paired with --input-jsonl, or standalone (requires "
        "--response-output or default path).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for input_data-style JSONL. Required when --input-jsonl is set.",
    )
    parser.add_argument(
        "--response-output",
        type=Path,
        default=None,
        help="Output for {\"prompt\", \"response\"} JSONL. Default: next to -o, or "
        "./input_response_data.jsonl when -o is omitted.",
    )
    parser.add_argument(
        "--key-start",
        type=int,
        default=0,
        help="Base key for each source row's first generation; within a row, keys are "
        "base + 0, base + 1, …; base advances by the number of generations per kept row.",
    )
    parser.add_argument(
        "--response-format",
        choices=("json", "text"),
        default="json",
        help="How to parse each line of --input-response-jsonl (default: json with text|response).",
    )
    parser.add_argument(
        "--response-count",
        choices=("auto", "one"),
        default="auto",
        help="When both inputs are set: auto = max(1, len(generations)) lines per kept row; "
        "one = single line per kept prompt row.",
    )
    args = parser.parse_args()

    if args.input_jsonl is None and args.input_response_jsonl is None:
        parser.error("Pass at least one of --input-jsonl or --input-response-jsonl.")
    if args.input_jsonl is not None and args.output is None:
        parser.error("--input-jsonl requires -o/--output.")
    if args.input_response_jsonl is not None and args.input_jsonl is None:
        if args.response_count != "auto":
            parser.error("--response-count applies only when both --input-jsonl and --input-response-jsonl are set.")
    if args.input_jsonl is None and args.response_count != "auto":
        parser.error("--response-count requires --input-jsonl together with --input-response-jsonl.")

    response_path = args.response_output or _default_response_output(args.output)

    if args.input_jsonl is not None and args.input_response_jsonl is not None:
        with args.input_jsonl.open(encoding="utf-8") as fin, args.input_response_jsonl.open(
            encoding="utf-8"
        ) as fin_resp, args.output.open("w", encoding="utf-8") as fout, response_path.open(
            "w", encoding="utf-8"
        ) as fout_resp:
            process_paired_postprocessed_and_responses(
                fin,
                fin_resp,
                fout,
                fout_resp,
                args.key_start,
                args.response_count,
                args.response_format,
            )
    elif args.input_jsonl is not None:
        with args.input_jsonl.open(encoding="utf-8") as fin, args.output.open(
            "w", encoding="utf-8"
        ) as fout:
            process_input_data_only(fin, fout, args.key_start)
    else:
        assert args.input_response_jsonl is not None
        with args.input_response_jsonl.open(encoding="utf-8") as fin, response_path.open(
            "w", encoding="utf-8"
        ) as fout_resp:
            process_response_data_only(fin, fout_resp, args.response_format)


if __name__ == "__main__":
    main()
