#!/usr/bin/env python3
"""Build A∪B toxicity failure union datasets with per-source tags."""

from __future__ import annotations

import json
from pathlib import Path

from laser_edit.ebm_training.build_failure_union_dataset import group_records_for_edit_input
from laser_edit.ebm_training.failure_union_utils import load_index_file
from laser_edit.utils.utils import read_outputs

SOURCE_A = "perspective_api_toxic"
SOURCE_B = "energy_fixed_0_95"


def infer_generations_per_prompt(jsonl_path: Path) -> int:
    with jsonl_path.open() as f:
        row = json.loads(f.readline())
    return len(row["generations"])


def load_perspective_toxic_indices(
    toxicity_path: Path,
    *,
    generations_per_prompt: int,
    toxic_threshold: float = 0.5,
) -> set[int]:
    indices: set[int] = set()
    with toxicity_path.open(encoding="utf-8") as f:
        for prompt_index, line in enumerate(f):
            if not line.strip():
                continue
            allresponses = json.loads(line)["allresponses"]
            for key, value in allresponses.items():
                generation_index = int(key.rsplit("-", 1)[-1])
                if value is None or value[0] is None:
                    continue
                prob = float(
                    value[0]["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
                )
                if prob > toxic_threshold:
                    indices.add(prompt_index * generations_per_prompt + generation_index)
    return indices


def build_union_toxicity_dataset(
    model_name: str,
    jsonl_path: Path,
    *,
    fixed_threshold: float = 0.95,
    fixed_threshold_label: str = "0_95",
    perspective_toxic_threshold: float = 0.5,
) -> tuple[list[dict], dict]:
    generations_per_prompt = infer_generations_per_prompt(jsonl_path)

    base = jsonl_path.name.rsplit(".jsonl", 1)[0]
    parent = jsonl_path.parent

    toxicity_path = jsonl_path.with_name(jsonl_path.name + "-results.txt.toxicity")
    b_index_path = parent / f"{base}_edit_candidates_{fixed_threshold_label}_index.txt"

    a = load_perspective_toxic_indices(
        toxicity_path,
        generations_per_prompt=generations_per_prompt,
        toxic_threshold=perspective_toxic_threshold,
    )
    b = load_index_file(b_index_path)
    union = sorted(a | b)

    flat = read_outputs(str(jsonl_path))
    if len(flat) % generations_per_prompt != 0:
        raise RuntimeError(
            f"Flat generation count {len(flat)} is not divisible by k={generations_per_prompt}"
        )
    expected_prompts = len(flat) // generations_per_prompt

    records: list[dict] = []
    for idx in union:
        row = flat.iloc[idx]
        sources: list[str] = []
        flags = {
            "perspective_api_toxic": idx in a,
            "energy_fixed_0_95": idx in b,
        }
        if flags["perspective_api_toxic"]:
            sources.append(SOURCE_A)
        if flags["energy_fixed_0_95"]:
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
            }
        )

    grouped_records = group_records_for_edit_input(records)

    summary = {
        "model": model_name,
        "source_jsonl": str(jsonl_path),
        "perspective_sidecar": str(toxicity_path),
        "universe_size": len(flat),
        "generations_per_prompt": generations_per_prompt,
        "prompt_count": expected_prompts,
        "thresholds": {
            "perspective_api_toxic": perspective_toxic_threshold,
            "energy_fixed_0_95": fixed_threshold,
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
