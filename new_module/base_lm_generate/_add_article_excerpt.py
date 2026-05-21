#!/usr/bin/env python3
"""Attach article excerpts to baselm ``comment/*.jsonl`` rows.

Adds ``prompt.article_excerpt`` from the SOC reference file and leaves ``prompt.text``
unchanged.

**SOC lookup key (matches reference ``comment_prefix``)**

- **Observed baselm data** (2026-05 scan of ``baselm_gens/**/comment/*.jsonl``): all 29
  files and every row use ``prompt.text`` only (continuation prefix). No row contained
  ``prompt.comment_prefix``. So lookup is normally ``prompt.text``.

- **`generate_main.py`**: the ``model_access_method == "vllm"`` branch writes the same
  shape: ``{"prompt": {"text": <comment_prefix>}}``. The non-``vllm`` branch would write
  ``comment_prefix`` plus ``text`` = article excerpt; for that hypothetical shape we
  must look up via ``prompt.comment_prefix`` (``text`` is not the prefix).

We therefore resolve the key as: ``comment_prefix`` if present, else ``text``.

Reference: ``new_module/data/nli-toxicity/socc_gnm_top250_ncomments_10sampled_root_comments.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE = (
    SCRIPT_DIR.parents[1]
    / "data"
    / "nli-toxicity"
    / "socc_gnm_top250_ncomments_10sampled_root_comments.jsonl"
)


def load_comment_prefix_to_excerpt(ref_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with ref_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prefix = row.get("comment_prefix")
            excerpt = row.get("excerpt")
            if not isinstance(prefix, str) or not isinstance(excerpt, str):
                continue
            if prefix in mapping and mapping[prefix] != excerpt:
                raise ValueError(
                    f"Ambiguous comment_prefix in reference (different excerpts): {prefix!r}"
                )
            mapping[prefix] = excerpt
    return mapping


def _comment_prefix_lookup_key(prompt: dict) -> str | None:
    """Key for SOC ``comment_prefix`` → ``excerpt`` map.

    Current baselm comment JSONLs only set ``prompt.text`` (prefix). If
    ``prompt.comment_prefix`` exists (e.g. non-``vllm`` ``generate_main`` output), use it
    because ``prompt.text`` is then the article excerpt, not the prefix.
    """
    cp = prompt.get("comment_prefix")
    if isinstance(cp, str) and cp.strip():
        return cp
    t = prompt.get("text")
    if isinstance(t, str) and t.strip():
        return t
    return None


def transform_record(obj: dict, mapping: dict[str, str]) -> tuple[dict, str | None]:
    """Return (new_obj, error_message). error_message set if excerpt could not be resolved."""
    prompt = obj.get("prompt")
    if not isinstance(prompt, dict):
        return obj, "missing or invalid prompt"

    key = _comment_prefix_lookup_key(prompt)
    if key is None:
        return obj, "need non-empty prompt.text (comment prefix), or prompt.comment_prefix if present"

    excerpt = mapping.get(key) or mapping.get(key.strip())
    if excerpt is None:
        prefix_preview = key[:80]
        return obj, f"no excerpt for comment_prefix ({len(key)} chars): {prefix_preview!r}…"

    if prompt.get("article_excerpt") == excerpt:
        return obj, None

    new_prompt = dict(prompt)
    new_prompt["article_excerpt"] = excerpt
    out = dict(obj)
    out["prompt"] = new_prompt
    return out, None


def _glob_files(base: Path, model_glob: str | None) -> list[Path]:
    root = base / "baselm_gens"
    paths: list[Path] = []
    for model_dir in sorted(root.glob(model_glob or "*")):
        if not model_dir.is_dir():
            continue
        cdir = model_dir / "comment"
        if not cdir.is_dir():
            continue
        paths.extend(sorted(cdir.glob("*.jsonl")))
    return paths


def process_paths(
    files: list[Path],
    mapping: dict[str, str],
    *,
    in_place: bool,
    dry_run: bool,
) -> tuple[int, int, list[tuple[Path, int, str]]]:
    updated_files = 0
    updated_lines = 0
    errors: list[tuple[Path, int, str]] = []

    for path in files:
        if not path.name.endswith(".jsonl"):
            continue
        rows_out: list[str] = []
        file_changed = False
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    rows_out.append(line)
                    continue
                obj = json.loads(line)
                new_obj, err = transform_record(obj, mapping)
                if err:
                    errors.append((path, lineno, err))
                    rows_out.append(json.dumps(obj, ensure_ascii=False))
                    continue
                if new_obj != obj:
                    file_changed = True
                    updated_lines += 1
                rows_out.append(json.dumps(new_obj, ensure_ascii=False))

        if not file_changed:
            continue
        updated_files += 1
        if dry_run:
            continue
        if not in_place:
            continue
        backup_path = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup_path)
        text = "\n".join(rows_out) + ("\n" if rows_out else "")
        path.write_text(text, encoding="utf-8")

    return updated_files, updated_lines, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="SOC comment JSONL with comment_prefix + excerpt.",
    )
    parser.add_argument(
        "--baselm-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory that contains ``baselm_gens/``.",
    )
    parser.add_argument(
        "--model-glob",
        type=str,
        default="*",
        help="Under baselm_gens/, only model directories matching this glob (default: all).",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit JSONL paths; if set, ignores --baselm-root / --model-glob.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite each file after copying the original to the same name with a ``.bak`` suffix (e.g. ``run.jsonl`` → ``run.jsonl.bak``).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write; report how many lines would change.",
    )
    args = parser.parse_args()

    ref_path = args.reference_jsonl.expanduser().resolve()
    if not ref_path.is_file():
        raise SystemExit(f"Reference file not found: {ref_path}")

    mapping = load_comment_prefix_to_excerpt(ref_path)

    if args.files:
        files = [p.expanduser().resolve() for p in args.files]
    else:
        files = _glob_files(args.baselm_root.expanduser().resolve(), args.model_glob)

    if not args.in_place and not args.dry_run:
        raise SystemExit("Choose --in-place to write files, or --dry-run to preview.")

    uf, ul, errs = process_paths(files, mapping, in_place=args.in_place, dry_run=args.dry_run)

    print(f"Files scanned: {len(files)}")
    print(f"Files with changes: {uf}")
    print(f"Lines updated: {ul}")
    if errs:
        print(f"Lines with unresolved prefix ({len(errs)}):")
        for p, ln, msg in errs[:20]:
            print(f"  {p}:{ln}: {msg}")
        if len(errs) > 20:
            print(f"  ... and {len(errs) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
