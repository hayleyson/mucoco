#!/usr/bin/env python3
"""
Aggregate min / max / mean / median sentence, word, and character counts over
generation files under ``baselm_gens``.

Each condition selects the **latest** ``*.jsonl`` in
``BASE_DIR / <model_slug> / <task> /`` whose basename matches
``<model_slug>_<task>_<prompt_type>_*.jsonl`` (few-shot runs include ``Nshot_``
before the timestamp, which this glob still covers).

CLI ``--prompt-suffixes`` are combined with a task-specific prefix: ``anli-r2-test``
→ ``nli_<suffix>``, ``nontoxic`` → ``nontoxic_<suffix>`` (e.g. ``plain`` →
``nli_plain`` / ``nontoxic_plain``).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

# Maps task folder name to the filename prefix used before ``_<suffix>`` in ``prompt_type``.
TASK_TO_PROMPT_PREFIX: dict[str, str] = {
    "nli": "nli",
    "nontoxic": "nontoxic",
    "nli_nontoxic": "nli_nontoxic",
    "comment": "comment",
}


def prompt_type_for_task(task: str, suffix: str) -> str:
    """Build full ``prompt_type`` for JSONL basenames, e.g. ``plain`` → ``nli_plain`` when task is anli."""
    prefix = TASK_TO_PROMPT_PREFIX.get(task)
    if prefix is None:
        known = ", ".join(sorted(TASK_TO_PROMPT_PREFIX))
        raise SystemExit(f"Unknown task {task!r}; add a mapping or use a known task: {known}")
    s = suffix.strip()
    if s.startswith("_"):
        s = s[1:]
    if not s:
        raise SystemExit("prompt suffix must be non-empty (e.g. plain, 0shot, few_shot_5shot).")
    return f"{prefix}_{s}"


def basename_from_hf_model(model: str) -> str:
    """Last path segment, e.g. ``openai/gpt-oss-20b`` -> ``gpt-oss-20b``."""
    return model.rstrip("/").split("/")[-1]


def _sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return len(parts) if parts else 1


def _word_count(text: str) -> int:
    return len(text.split())


def _char_count(text: str) -> int:
    return len(text)


def iter_generation_texts(jsonl_path: Path) -> Iterable[str]:
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for gen in obj.get("generations", []):
                t = gen.get("text")
                if t is not None:
                    yield str(t)


def find_latest_jsonl(base_dir: Path, model_slug: str, task: str, prompt_type: str) -> Path | None:
    sub = base_dir / model_slug / task
    if not sub.is_dir():
        return None
    prefix = f"{model_slug}_{task}_{prompt_type}_"
    matches = sorted(
        [p for p in sub.glob("*.jsonl") if p.name.startswith(prefix)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
        }
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
    }


def stats_for_jsonl(path: Path) -> tuple[int, dict[str, dict[str, float]]]:
    """Per-metric min/max/mean/median across every ``generations[].text`` in the file."""
    sent: list[float] = []
    words: list[float] = []
    chars: list[float] = []
    num = 0
    for t in iter_generation_texts(path):
        num += 1
        sent.append(float(_sentence_count(t)))
        words.append(float(_word_count(t)))
        chars.append(float(_char_count(t)))
    metrics = {
        "sentence_count": summarize(sent),
        "word_count": summarize(words),
        "char_count": summarize(chars),
    }
    return num, metrics


def _float_csv(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def row_for_condition(
    base_dir: Path,
    model_slug: str,
    task: str,
    prompt_type: str,
) -> dict[str, Any]:
    path = find_latest_jsonl(base_dir, model_slug, task, prompt_type)
    if path is None:
        return {
            "model_slug": model_slug,
            "task": task,
            "prompt_type": prompt_type,
            "jsonl_path": "",
            "num_generations": 0,
            "sentence_min": "",
            "sentence_max": "",
            "sentence_mean": "",
            "sentence_median": "",
            "word_min": "",
            "word_max": "",
            "word_mean": "",
            "word_median": "",
            "char_min": "",
            "char_max": "",
            "char_mean": "",
            "char_median": "",
        }
    n, metrics = stats_for_jsonl(path)
    row: dict[str, Any] = {
        "model_slug": model_slug,
        "task": task,
        "prompt_type": prompt_type,
        "jsonl_path": str(path.resolve()),
        "num_generations": n,
    }
    for metric_key, prefixes in (
        ("sentence_count", ("sentence_",)),
        ("word_count", ("word_",)),
        ("char_count", ("char_",)),
    ):
        s = metrics[metric_key]
        p = prefixes[0]
        row[f"{p}min"] = _float_csv(s["min"])
        row[f"{p}max"] = _float_csv(s["max"])
        row[f"{p}mean"] = _float_csv(s["mean"])
        row[f"{p}median"] = _float_csv(s["median"])
    return row


def expand_product(
    models: Sequence[str],
    tasks: Sequence[str],
    prompt_suffixes: Sequence[str],
) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for m, t, suf in itertools.product(models, tasks, prompt_suffixes):
        out.append((m, t, prompt_type_for_task(t, suf)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize lengths in latest generation jsonl files.")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "baselm_gens",
        help="Root directory (same layout as generate_main_local.sh FILE_SAVE_DIR parent).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the summary CSV.",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Model basename(s) matching folder names (e.g. gpt-oss-20b Qwen3-8B).",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=[],
        help="Task subfolder names (e.g. anli-r2-test nontoxic).",
    )
    p.add_argument(
        "--prompt-suffixes",
        nargs="*",
        default=[],
        dest="prompt_suffixes",
        help=(
            "Suffix only (after task prefix): plain, 0shot, few_shot_5shot. "
            "Resolved to nli_* for anli-r2-test and nontoxic_* for nontoxic."
        ),
    )
    return p.parse_args()


# Edit this list for fixed condition sets without CLI args. Use full ``prompt_type``
# strings (as in filenames), e.g. ``("gpt-oss-20b", "anli-r2-test", "nli_plain")``.
DEFAULT_CONDITIONS: list[tuple[str, str, str]] = [
    # ("gpt-oss-20b", "anli-r2-test", "nli_plain"),
]

CSV_FIELDNAMES = [
    "model_slug",
    "task",
    "prompt_type",
    "jsonl_path",
    "num_generations",
    "sentence_min",
    "sentence_max",
    "sentence_mean",
    "sentence_median",
    "word_min",
    "word_max",
    "word_mean",
    "word_median",
    "char_min",
    "char_max",
    "char_mean",
    "char_median",
]


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if args.models or args.tasks or args.prompt_suffixes:
        if not (args.models and args.tasks and args.prompt_suffixes):
            raise SystemExit(
                "Provide all of --models, --tasks, and --prompt-suffixes (Cartesian product; "
                "suffixes are expanded per task)."
            )
        conditions = expand_product(args.models, args.tasks, args.prompt_suffixes)
    else:
        conditions = list(DEFAULT_CONDITIONS)

    if not conditions:
        raise SystemExit(
            "No conditions: pass --models, --tasks, and --prompt-suffixes, "
            "or fill DEFAULT_CONDITIONS in this script (full prompt_type per row)."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_slug, task, prompt_type in conditions:
        rows.append(row_for_condition(base_dir, model_slug, task, prompt_type))

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDNAMES})

    print(f"Wrote {len(rows)} rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
