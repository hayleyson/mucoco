#!/usr/bin/env python3
"""Build A∪B failure union datasets with per-source tags."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from laser_edit.ebm_training.failure_union_utils import (
    FILES,
    load_index_file,
    load_nli_contradiction_indices,
)
from laser_edit.utils.utils import read_outputs

ROOT = Path("/home/hyeryung/data/mucoco")

SOURCE_A = "external_nli_contradiction"
SOURCE_B = "energy_fixed_0_99"
SOURCE_C = "energy_val_best_f1"


def infer_generations_per_prompt(jsonl_path: Path) -> int:
    with jsonl_path.open() as f:
        row = json.loads(f.readline())
    return len(row["generations"])


def save_nli_contradiction_edit_candidates(jsonl_path: Path) -> tuple[Path, Path, dict]:
    """Save edit candidates for 3-model NLI ensemble contradictions only."""
    generations_per_prompt = infer_generations_per_prompt(jsonl_path)
    base = jsonl_path.name.rsplit(".jsonl", 1)[0]
    parent = jsonl_path.parent
    nli_path = jsonl_path.with_name(jsonl_path.name + "-results.txt.nli")
    if not nli_path.exists():
        raise FileNotFoundError(f"Missing NLI sidecar: {nli_path}")

    indices = sorted(load_nli_contradiction_indices(nli_path))
    flat = read_outputs(str(jsonl_path))
    nli_rows = load_nli_rows(nli_path)
    if len(flat) != len(nli_rows):
        raise RuntimeError(
            f"Row count mismatch: generations={len(flat)} nli={len(nli_rows)}"
        )

    records: list[dict] = []
    for idx in indices:
        row = flat.iloc[idx]
        nli = nli_rows[idx]
        records.append(
            {
                "raveled_index": idx,
                "premise_index": idx // generations_per_prompt,
                "generation_index": idx % generations_per_prompt,
                "prompt": {"text": row["prompt"]},
                "generation": {"text": row["text"]},
                "failure_sources": [SOURCE_A],
                "failure_source_flags": {"external_nli_contradiction": True},
                "external_nli": {
                    "nli_class": nli.get("nli_class"),
                    "entailment_prob": nli.get("entailment_prob"),
                    "neutral_prob": nli.get("neutral_prob"),
                    "contradiction_prob": nli.get("contradiction_prob"),
                },
            }
        )

    grouped = group_records_for_edit_input(records)
    out_path = parent / f"{base}_edit_candidates_nli_contradiction.jsonl"
    index_path = parent / f"{base}_edit_candidates_nli_contradiction_index.txt"
    with out_path.open("w") as f:
        for rec in grouped:
            f.write(json.dumps(rec) + "\n")
    index_path.write_text(" ".join(str(i) for i in indices) + "\n")

    summary = {
        "source_jsonl": str(jsonl_path),
        "nli_sidecar": str(nli_path),
        "output_generation_count": len(indices),
        "output_prompt_count": len(grouped),
        "output_format": "prompt_generations",
    }
    return out_path, index_path, summary


def group_records_for_edit_input(flat_records: list[dict]) -> list[dict]:
    """Group flat failure records into edit-pipeline JSONL rows.

    Each output row matches ``edit_candidates`` / ``ebm_edit_main`` input:
    ``{"prompt": {"text": ...}, "generations": [{"text": ...}, ...]}``.
    Per-generation audit fields are preserved on each generation dict.
    """
    by_premise: dict[int, dict] = {}
    for rec in flat_records:
        premise_index = rec["premise_index"]
        generation = {
            "text": rec["generation"]["text"],
            "raveled_index": rec["raveled_index"],
            "generation_index": rec["generation_index"],
            "failure_sources": rec["failure_sources"],
            "failure_source_flags": rec["failure_source_flags"],
        }
        if "external_nli" in rec:
            generation["external_nli"] = rec["external_nli"]
        if "perspective_toxicity" in rec:
            generation["perspective_toxicity"] = rec["perspective_toxicity"]
        if premise_index not in by_premise:
            by_premise[premise_index] = {
                "prompt": rec["prompt"],
                "generations": [],
            }
        by_premise[premise_index]["generations"].append(generation)

    grouped: list[dict] = []
    for premise_index in sorted(by_premise):
        row = by_premise[premise_index]
        row["generations"].sort(key=lambda g: g["generation_index"])
        grouped.append(row)
    return grouped


def load_nli_rows(nli_path: Path) -> list[dict]:
    rows: list[dict] = []
    with nli_path.open() as f:
        for line in f:
            rows.append(ast.literal_eval(line.strip()))
    return rows


def build_union_dataset(model_name: str, jsonl_path: Path) -> tuple[list[dict], dict]:
    generations_per_prompt = infer_generations_per_prompt(jsonl_path)

    base = jsonl_path.name.rsplit(".jsonl", 1)[0]
    parent = jsonl_path.parent

    nli_path = jsonl_path.with_name(jsonl_path.name + "-results.txt.nli")
    b_index_path = parent / f"{base}_edit_candidates_0_99_index.txt"

    a = load_nli_contradiction_indices(nli_path)
    b = load_index_file(b_index_path)
    union = sorted(a | b)

    flat = read_outputs(str(jsonl_path))
    nli_rows = load_nli_rows(nli_path)
    if len(flat) != len(nli_rows):
        raise RuntimeError(
            f"Row count mismatch for {model_name}: generations={len(flat)} nli={len(nli_rows)}"
        )

    records: list[dict] = []
    for idx in union:
        row = flat.iloc[idx]
        nli = nli_rows[idx]
        sources: list[str] = []
        flags = {
            "external_nli_contradiction": idx in a,
            "energy_fixed_0_99": idx in b,
        }
        if flags["external_nli_contradiction"]:
            sources.append(SOURCE_A)
        if flags["energy_fixed_0_99"]:
            sources.append(SOURCE_B)

        records.append(
            {
                "model": model_name,
                "raveled_index": idx,
                "premise_index": idx // generations_per_prompt,
                "generation_index": idx % generations_per_prompt,
                "prompt": {"text": row["prompt"]},
                "generation": {"text": row["text"]},
                "failure_sources": sources,
                "failure_source_flags": flags,
                "external_nli": {
                    "nli_class": nli.get("nli_class"),
                    "entailment_prob": nli.get("entailment_prob"),
                    "neutral_prob": nli.get("neutral_prob"),
                    "contradiction_prob": nli.get("contradiction_prob"),
                },
            }
        )

    grouped_records = group_records_for_edit_input(records)

    summary = {
        "model": model_name,
        "source_jsonl": str(jsonl_path),
        "universe_size": len(flat),
        "generations_per_prompt": generations_per_prompt,
        "thresholds": {
            "energy_fixed_0_99": 0.99,
        },
        "subset_sizes": {
            SOURCE_A: len(a),
            SOURCE_B: len(b),
            "union": len(union),
        },
        "output_generation_count": len(records),
        "output_prompt_count": len(grouped_records),
        "output_format": "prompt_generations",
    }
    return grouped_records, summary


def build_nli_abc_union_dataset(
    model_name: str,
    jsonl_path: Path,
    fixed_index_path: Path,
    best_f1_index_path: Path,
    best_f1_threshold: float,
) -> tuple[list[dict], dict]:
    """Build an A∪B∪C NLI failure union in the established edit-input schema."""
    generations_per_prompt = infer_generations_per_prompt(jsonl_path)
    nli_path = jsonl_path.with_name(jsonl_path.name + "-results.txt.nli")

    a = load_nli_contradiction_indices(nli_path)
    b = load_index_file(fixed_index_path)
    c = load_index_file(best_f1_index_path)
    union = sorted(a | b | c)

    flat = read_outputs(str(jsonl_path))
    nli_rows = load_nli_rows(nli_path)
    if len(flat) != len(nli_rows):
        raise RuntimeError(
            f"Row count mismatch for {model_name}: generations={len(flat)} nli={len(nli_rows)}"
        )

    records: list[dict] = []
    for idx in union:
        row = flat.iloc[idx]
        nli = nli_rows[idx]
        flags = {
            SOURCE_A: idx in a,
            SOURCE_B: idx in b,
            SOURCE_C: idx in c,
        }
        records.append(
            {
                "model": model_name,
                "raveled_index": idx,
                "premise_index": idx // generations_per_prompt,
                "generation_index": idx % generations_per_prompt,
                "prompt": {"text": row["prompt"]},
                "generation": {"text": row["text"]},
                "failure_sources": [source for source, active in flags.items() if active],
                "failure_source_flags": flags,
                "external_nli": {
                    "nli_class": nli.get("nli_class"),
                    "entailment_prob": nli.get("entailment_prob"),
                    "neutral_prob": nli.get("neutral_prob"),
                    "contradiction_prob": nli.get("contradiction_prob"),
                },
            }
        )

    grouped_records = group_records_for_edit_input(records)
    summary = {
        "model": model_name,
        "source_jsonl": str(jsonl_path),
        "nli_sidecar": str(nli_path),
        "universe_size": len(flat),
        "generations_per_prompt": generations_per_prompt,
        "thresholds": {
            SOURCE_B: 0.99,
            SOURCE_C: best_f1_threshold,
        },
        "subset_sizes": {
            SOURCE_A: len(a),
            SOURCE_B: len(b),
            SOURCE_C: len(c),
            "union": len(union),
        },
        "output_generation_count": len(records),
        "output_prompt_count": len(grouped_records),
        "output_format": "prompt_generations",
    }
    return grouped_records, summary


def main() -> None:
    all_summaries = []
    for model_name, jsonl_path in FILES.items():
        records, summary = build_union_dataset(model_name, jsonl_path)
        base = jsonl_path.name.rsplit(".jsonl", 1)[0]
        out_path = jsonl_path.with_name(f"{base}_failure_union_ab.jsonl")
        summary_path = jsonl_path.with_name(f"{base}_failure_union_ab.summary.json")

        with out_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        all_summaries.append(summary)
        print(
            f"Wrote {summary['output_prompt_count']} prompts "
            f"({summary['output_generation_count']} generations) to {out_path}"
        )

    combined_summary_path = ROOT / "laser_edit/ebm_training/failure_union_ab.summary.json"
    combined_summary_path.write_text(json.dumps(all_summaries, indent=2) + "\n")
    print(f"Wrote combined summary to {combined_summary_path}")


if __name__ == "__main__":
    main()
