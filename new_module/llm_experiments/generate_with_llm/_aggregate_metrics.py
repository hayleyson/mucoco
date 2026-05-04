#!/usr/bin/env python3
"""
For each (model, task, prompt_suffix), pick the **latest** matching ``*.jsonl`` under
``baselm_gens/<model>/<task>/`` (same rules as ``_analyze_model_gen_lengths.py``),
read the companion ``<file>.jsonl-results.txt``, parse ``key: value`` entries (comma-
separated on each line), and write one Excel sheet per task.

See ``_analyze_model_gen_lengths.py`` for filename / ``prompt_type`` construction.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from openpyxl import Workbook  # type: ignore[import-untyped]
except ImportError as e:
    raise SystemExit("openpyxl is required: pip install openpyxl") from e


def _load_analyze_helpers() -> Any:
    path = Path(__file__).resolve().parent / "_analyze_model_gen_lengths.py"
    spec = importlib.util.spec_from_file_location("_analyze_model_gen_lengths", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_aml = _load_analyze_helpers()
find_latest_jsonl = _aml.find_latest_jsonl
prompt_type_for_task = _aml.prompt_type_for_task
expand_product = _aml.expand_product


def results_path_for_jsonl(jsonl_path: Path) -> Path:
    return Path(str(jsonl_path) + "-results.txt")


def parse_results_text(text: str) -> dict[str, str]:
    """Merge ``key: value`` pairs from all non-empty lines (comma-separated per line)."""
    merged: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in line.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = key.strip()
            val = val.strip()
            if key:
                merged[key] = val
    return merged


def read_results_file(path: Path) -> tuple[dict[str, str], list[str]]:
    raw_lines = []
    merged: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        s = line.strip()
        if s:
            raw_lines.append(s)
        merged.update(parse_results_text(line))
    return merged, raw_lines


def excel_sheet_title(task: str) -> str:
    # Excel worksheet name ≤ 31 chars; no : \ / ? * [ ]
    t = task[:31]
    for bad in (":", "\\", "/", "?", "*", "[", "]"):
        t = t.replace(bad, "_")
    return t or "task"


def sanitize_col(name: str) -> str:
    return name.strip()[:255]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate metrics from *.jsonl-results.txt into one xlsx sheet per task."
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "baselm_gens",
        help="Root directory (same as _analyze_model_gen_lengths).",
    )
    p.add_argument(
        "--output-xlsx",
        type=Path,
        required=True,
        help="Path for the output .xlsx file.",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Model basename(s) matching folder names under base-dir.",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=[],
        help="Task subfolder names (e.g. nli nontoxic).",
    )
    p.add_argument(
        "--prompt-suffixes",
        nargs="*",
        default=[],
        dest="prompt_suffixes",
        help="Prompt suffix(es) resolved per task (see _analyze_model_gen_lengths).",
    )
    return p.parse_args()


# Fixed condition triples without CLI: (model_slug, task_folder, full prompt_type).
DEFAULT_CONDITIONS: list[tuple[str, str, str]] = []


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if args.models or args.tasks or args.prompt_suffixes:
        if not (args.models and args.tasks and args.prompt_suffixes):
            raise SystemExit(
                "Provide all of --models, --tasks, and --prompt-suffixes (Cartesian product), "
                "or fill DEFAULT_CONDITIONS."
            )
        conditions = expand_product(args.models, args.tasks, args.prompt_suffixes)
        task_order = list(dict.fromkeys(args.tasks))
    else:
        conditions = list(DEFAULT_CONDITIONS)
        task_order = list(dict.fromkeys(t for _, t, _ in conditions))

    if not conditions:
        raise SystemExit(
            "No conditions: pass --models, --tasks, --prompt-suffixes "
            "or populate DEFAULT_CONDITIONS."
        )

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for model_slug, task, prompt_type in conditions:
        jsonl = find_latest_jsonl(base_dir, model_slug, task, prompt_type)
        row: dict[str, Any] = {
            "model_slug": model_slug,
            "prompt_type": prompt_type,
            "jsonl_path": str(jsonl.resolve()) if jsonl else "",
            "results_path": "",
            "jsonl_found": jsonl is not None,
            "results_found": False,
            "metric_lines_raw": "",
        }
        merged: dict[str, str] = {}
        raw_lines: list[str] = []
        if jsonl is not None:
            rpath = results_path_for_jsonl(jsonl)
            row["results_path"] = str(rpath.resolve())
            if rpath.is_file():
                merged, raw_lines = read_results_file(rpath)
                row["results_found"] = True
        row["metric_lines_raw"] = "\n".join(raw_lines)
        row.update({sanitize_col(k): v for k, v in merged.items()})
        rows_by_task[task].append(row)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    first_sheet = True
    sheets_seen: set[str] = set()

    for task in task_order:
        if task not in rows_by_task:
            continue
        rows = rows_by_task[task]

        metric_keys_for_sheet: set[str] = set()
        for r in rows:
            for k in r:
                if k in (
                    "model_slug",
                    "prompt_type",
                    "jsonl_path",
                    "results_path",
                    "jsonl_found",
                    "results_found",
                    "metric_lines_raw",
                ):
                    continue
                metric_keys_for_sheet.add(k)

        metric_cols = sorted(metric_keys_for_sheet)
        fixed = [
            "model_slug",
            "prompt_type",
            "jsonl_path",
            "results_path",
            "jsonl_found",
            "results_found",
            "metric_lines_raw",
        ]
        columns = fixed + metric_cols

        title = excel_sheet_title(task)
        original = title
        n = 1
        while title in sheets_seen:
            suffix = f"_{n}"
            title = excel_sheet_title(f"{original[: 31 - len(suffix)]}{suffix}")
            n += 1
        sheets_seen.add(title)

        if first_sheet:
            ws = wb.active
            ws.title = title
            first_sheet = False
        else:
            ws = wb.create_sheet(title)

        ws.append(columns)
        for r in rows:
            ws.append([r.get(c, "") for c in columns])

    if first_sheet:
        raise SystemExit("No data rows matched the given tasks/models.")

    wb.save(args.output_xlsx)
    n_rows = sum(len(rows_by_task[t]) for t in task_order if t in rows_by_task)
    print(f"Wrote {n_rows} row(s) across {len(sheets_seen)} sheet(s) to {args.output_xlsx.resolve()}")


if __name__ == "__main__":
    main()
