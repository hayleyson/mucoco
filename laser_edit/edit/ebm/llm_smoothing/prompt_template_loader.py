"""Load grammar-refinement templates (prompt_templates.json) and few-shot example files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

DEFAULT_PROMPT_TEMPLATES_PATH = Path(__file__).resolve().parent / "prompt_templates.json"

_json_cache: dict[str, Any] = {}


def _load_json(path: Path) -> Any:
    key = str(path.resolve())
    if key not in _json_cache:
        with open(path, encoding="utf-8") as f:
            _json_cache[key] = json.load(f)
    return _json_cache[key]


def load_grammar_refinement_prompt_templates(
    path: Optional[Union[str, Path]] = None,
    template_version: Optional[Union[int, str]] = None,
) -> dict[str, str]:
    """Headers, footers, examples_section_prefix, system_message for TextRefiner."""
    path = Path(path) if path is not None else DEFAULT_PROMPT_TEMPLATES_PATH
    data = _load_json(path)
    if not template_version:
        raise ValueError(f"template_version is required")
    ver = str(template_version)
    if ver not in data["grammar_refinement_versions"]:
        raise ValueError(f"template_version {ver} not found in {path}")
    block = data["grammar_refinement_versions"][ver]
    return {
        "examples_section_prefix": data.get("examples_section_prefix", "\n### EXAMPLES\n"),
        "sentence_header": block["sentence"]["header"],
        "sentence_footer": block["sentence"]["footer"],
        "continuation_header": block["continuation"]["header"],
        "continuation_footer": block["continuation"]["footer"],
    }

def load_few_shot_examples(
    path: Union[str, Path],
    task: str,
    num_shots: int,
) -> list[dict[str, Any]]:
    """Few-shot dicts for task / mode (flat `{task: [...]}` or legacy nested)."""
    path = Path(path)
    raw = _load_json(path)
    examples = raw[task]
    n = min(num_shots, len(examples))
    if n < num_shots:
        print(
            f"Warning: requested {num_shots} shots but only {len(examples)} for {task!r}; using {n}."
        )
    return [dict(ex) for ex in examples[:n]]
