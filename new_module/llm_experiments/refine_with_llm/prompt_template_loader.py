"""Load refine / edit prompt templates from prompt_templates.json (extensible, versioned)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

DEFAULT_PROMPT_TEMPLATES_PATH = Path(__file__).resolve().parent / "prompt_templates.json"

_file_cache: dict[str, dict[str, Any]] = {}


def clear_prompt_template_cache() -> None:
    """Drop cached JSON (e.g. after tests edit the file on disk)."""
    _file_cache.clear()


def load_prompt_templates_file(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Parse prompt_templates.json. Results are cached per resolved path."""
    path = Path(path) if path is not None else DEFAULT_PROMPT_TEMPLATES_PATH
    key = str(path.resolve())
    if key not in _file_cache:
        with open(path, "r", encoding="utf-8") as f:
            _file_cache[key] = json.load(f)
    return _file_cache[key]


def _coerce_version_key(version: Union[int, str]) -> str:
    return str(version)


def resolve_grammar_refinement_version(
    data: Mapping[str, Any],
    template_version: Optional[Union[int, str]] = None,
) -> str:
    """Pick which grammar_refinement_versions key to use."""
    if template_version is not None:
        return _coerce_version_key(template_version)
    defaults = data.get("defaults") or {}
    return _coerce_version_key(defaults.get("grammar_refinement_version", "2"))


def _footer_for_mode(cfg: Mapping[str, str], footer_mode: str) -> str:
    if footer_mode == "sentence":
        return cfg["sentence_footer"]
    if footer_mode == "continuation":
        return cfg["continuation_footer"]
    raise ValueError(f"footer_mode must be 'sentence' or 'continuation', got {footer_mode!r}")


def get_edit_prompt_entry(
    prompt_type: str,
    *,
    path: Optional[Union[str, Path]] = None,
    data: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    """Return edit_prompts[prompt_type] as {header, footer_mode}."""
    raw = data if data is not None else load_prompt_templates_file(path)
    tasks = raw.get("edit_prompts")
    if not isinstance(tasks, dict):
        raise KeyError("prompt_templates.json missing edit_prompts object")
    if prompt_type not in tasks:
        raise KeyError(
            f"edit_prompts has no {prompt_type!r}. Available: {sorted(tasks)}"
        )
    entry = tasks[prompt_type]
    if isinstance(entry, str):
        raise TypeError(
            f"edit_prompts[{prompt_type!r}] must be an object with "
            f'"header" and "footer_mode", not a string'
        )
    try:
        header = entry["header"]
        footer_mode = entry["footer_mode"]
    except KeyError as e:
        raise KeyError(f"edit_prompts[{prompt_type!r}] missing key: {e}") from e
    if footer_mode not in ("sentence", "continuation"):
        raise ValueError(
            f"edit_prompts[{prompt_type!r}].footer_mode must be "
            f"'sentence' or 'continuation', got {footer_mode!r}"
        )
    return {"header": header, "footer_mode": footer_mode}


def build_edit_prompt_template(
    prompt_type: str,
    *,
    template_version: Optional[Union[int, str]] = None,
    path: Optional[Union[str, Path]] = None,
) -> str:
    """Instruction header from edit_prompts + shared grammar footer for that footer_mode."""
    p = Path(path) if path is not None else DEFAULT_PROMPT_TEMPLATES_PATH
    data = load_prompt_templates_file(p)
    entry = get_edit_prompt_entry(prompt_type, data=data)
    cfg = load_grammar_refinement_config(p, template_version=template_version)
    return entry["header"] + _footer_for_mode(cfg, entry["footer_mode"])


def load_grammar_refinement_config(
    path: Optional[Union[str, Path]] = None,
    template_version: Optional[Union[int, str]] = None,
) -> dict[str, str]:
    """Flatten grammar refinement fields for TextRefiner (system message, headers, footers)."""
    data = load_prompt_templates_file(path)
    try:
        system_message = data["system_message"]
        versions: dict[str, Any] = data["grammar_refinement_versions"]
    except KeyError as e:
        raise KeyError(f"prompt_templates.json missing required key: {e}") from e

    ver = resolve_grammar_refinement_version(data, template_version)
    if ver not in versions:
        raise KeyError(
            f"grammar_refinement_versions has no {ver!r}. "
            f"Available: {sorted(versions.keys(), key=lambda x: str(x))}"
        )
    block = versions[ver]
    examples_section_prefix = data.get("examples_section_prefix", "\n### EXAMPLES\n")

    out: dict[str, str] = {
        "system_message": system_message,
        "examples_section_prefix": examples_section_prefix,
    }
    for mode in ("sentence", "continuation"):
        if mode not in block:
            raise KeyError(
                f"grammar_refinement_versions[{ver!r}] missing mode {mode!r}"
            )
        part = block[mode]
        for key in ("header", "footer"):
            if key not in part:
                raise KeyError(
                    f"grammar_refinement_versions[{ver!r}][{mode!r}] "
                    f"missing {key!r}"
                )
        out[f"{mode}_header"] = part["header"]
        out[f"{mode}_footer"] = part["footer"]
    return out


def get_task_edit_prompt(
    prompt_type: str,
    *,
    template_version: Optional[Union[int, str]] = None,
    path: Optional[Union[str, Path]] = None,
) -> str:
    """Full user prompt: edit instruction header + versioned sentence/continuation footer."""
    return build_edit_prompt_template(
        prompt_type, template_version=template_version, path=path
    )
