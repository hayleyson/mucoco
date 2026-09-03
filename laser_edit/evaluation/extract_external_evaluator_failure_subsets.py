#!/usr/bin/env python3
"""Extract original GPT-3.5 external-evaluator failure subsets and saved metrics."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/hyeryung/data/mucoco")
TOX_SOURCE = ROOT / (
    "laser_edit/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nontoxic/"
    "gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl"
)
NLI_SOURCE = ROOT / "laser_edit/data/logical-consistency/anli-r2-test_prompt_4.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def external_toxicity_indices(toxicity_path: Path, *, generations_per_prompt: int) -> list[int]:
    indices = []
    for prompt_index, line in enumerate(toxicity_path.read_text(encoding="utf-8").splitlines()):
        responses = json.loads(line)["allresponses"]
        for key, value in responses.items():
            if value is None or value[0] is None:
                continue
            score = value[0]["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            if float(score) > 0.5:
                indices.append(prompt_index * generations_per_prompt + int(key.rsplit("-", 1)[-1]))
    return sorted(indices)


def external_nli_indices(nli_path: Path) -> list[int]:
    return [
        index
        for index, line in enumerate(nli_path.read_text(encoding="utf-8").splitlines())
        if ast.literal_eval(line)["nli_class"] == "contradiction"
    ]


def extract_lines(lines: list[str], indices: list[int]) -> list[str]:
    if not indices or max(indices) >= len(lines):
        raise ValueError(f"sidecar has {len(lines)} rows but requires index {max(indices, default=-1)}")
    return [lines[index] for index in indices]


def grouped_subset(source_rows: list[dict], indices: list[int]) -> list[dict]:
    flattened = [
        (prompt_index, generation_index, row["prompt"], generation)
        for prompt_index, row in enumerate(source_rows)
        for generation_index, generation in enumerate(row["generations"])
    ]
    grouped: dict[int, dict] = {}
    for index in indices:
        prompt_index, generation_index, prompt, generation = flattened[index]
        if prompt_index not in grouped:
            grouped[prompt_index] = {"prompt": prompt, "generations": []}
        grouped[prompt_index]["generations"].append(generation)
    return [grouped[prompt_index] for prompt_index in sorted(grouped)]


def distinctness(rows: list[dict]) -> tuple[float, float, float]:
    scores = [[], [], []]
    for row in rows:
        words = [generation["text"].split(" ") for generation in row["generations"]]
        total = sum(len(item) for item in words)
        grams = [set(), set(), set()]
        for tokens in words:
            grams[0].update(tokens)
            grams[1].update(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
            grams[2].update(f"{tokens[i]}_{tokens[i + 1]}_{tokens[i + 2]}" for i in range(len(tokens) - 2))
        for order in range(3):
            scores[order].append(len(grams[order]) / total)
    return tuple(float(np.mean(values)) for values in scores)


def ppl_metrics(lines: list[str]) -> tuple[float, float]:
    values = [[float(item.strip()) for item in line.split(",")] for line in lines]
    return float(np.mean([item[0] for item in values])), math.exp(sum(item[1] for item in values) / sum(item[2] for item in values))


def write_common_sidecars(output_base: Path, source_base: Path, indices: list[int], suffixes: list[str]) -> dict[str, list[str]]:
    extracted = {}
    for suffix in suffixes:
        source = Path(f"{source_base}-results.txt.{suffix}")
        lines = extract_lines(source.read_text(encoding="utf-8").splitlines(), indices)
        Path(f"{output_base}-results.txt.{suffix}").write_text("\n".join(lines) + "\n", encoding="utf-8")
        extracted[suffix] = lines
    return extracted


def write_toxicity_sidecar(output_base: Path, source_sidecar: Path, indices: list[int], k: int) -> list[str]:
    selected = set(indices)
    lines = []
    for prompt_index, line in enumerate(source_sidecar.read_text(encoding="utf-8").splitlines()):
        responses = json.loads(line)["allresponses"]
        subset = {
            key: value
            for key, value in responses.items()
            if prompt_index * k + int(key.rsplit("-", 1)[-1]) in selected
        }
        if subset:
            lines.append(json.dumps({"allresponses": subset}, ensure_ascii=False))
    Path(f"{output_base}-results.txt.toxicity").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def toxicity_metrics(rows: list[dict], toxicity_lines: list[str], common: dict[str, list[str]]) -> dict:
    generation_scores, prompt_maxes = [], []
    for line in toxicity_lines:
        scores = [
            float(value[0]["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
            for value in json.loads(line)["allresponses"].values()
        ]
        generation_scores.extend(scores)
        prompt_maxes.append(max(scores))
    d1, d2, d3 = distinctness(rows)
    mean_ppl, total_ppl = ppl_metrics(common["ppl-big-qwen"])
    fluent = [line.split(",", 1)[0] == "LABEL_1" for line in common["fluency"]]
    return {
        "sample_count": len(generation_scores),
        "prompt_count": len(rows),
        "avg_max_toxicity": float(np.mean(prompt_maxes)),
        "toxic_probability_p": float(np.mean([value > 0.5 for value in prompt_maxes])),
        "avg_toxicity": float(np.mean(generation_scores)),
        "toxic_probability_s": float(np.mean([value > 0.5 for value in generation_scores])),
        "constraint_satisfaction": float(np.mean([value <= 0.5 for value in generation_scores])),
        "ppl_qwen": mean_ppl,
        "total_ppl_qwen": total_ppl,
        "dist-1": d1,
        "dist-2": d2,
        "dist-3": d3,
        "repetition_rate": float(np.mean([bool(ast.literal_eval(line)) for line in common["repetitions"]])),
        "fluent_proba": float(np.mean(fluent)),
    }


def nli_metrics(rows: list[dict], common: dict[str, list[str]]) -> dict:
    nli = [ast.literal_eval(line) for line in common["nli"]]
    d1, d2, d3 = distinctness(rows)
    mean_ppl, total_ppl = ppl_metrics(common["ppl-big-qwen"])
    fluency = [line.split(",", 1)[0] == "LABEL_1" for line in common["fluency"]]
    return {
        "sample_count": len(nli),
        "prompt_count": len(rows),
        "avg_nli_entail": float(np.mean([row["entailment_prob"] for row in nli])),
        "avg_nli_neutral": float(np.mean([row["neutral_prob"] for row in nli])),
        "avg_nli_contradiction": float(np.mean([row["contradiction_prob"] for row in nli])),
        "contradiction_proba": float(np.mean([row["nli_class"] == "contradiction" for row in nli])),
        "constraint_satisfaction": float(np.mean([row["nli_class"] != "contradiction" for row in nli])),
        "ppl_qwen": mean_ppl,
        "total_ppl_qwen": total_ppl,
        "dist-1": d1,
        "dist-2": d2,
        "dist-3": d3,
        "repetition_rate": float(np.mean([bool(ast.literal_eval(line)) for line in common["repetitions"]])),
        "fluent_proba": float(np.mean(fluency)),
    }


def write_reports(output_dir: Path, task: str, metrics: dict, manifest: dict) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "results.txt").write_text("\n".join(f"{key}: {value}" for key, value in metrics.items()) + "\n", encoding="utf-8")
    columns = ["Ctrl. Acc.", "PPL", "Delta PPL", "Dist-3", "BERTScore", "Toks/s"]
    if task == "toxicity":
        columns.insert(1, "Avg. Toxicity")
    with (output_dir / "paper_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        row = {
            "Ctrl. Acc.": metrics["constraint_satisfaction"],
            "PPL": metrics["total_ppl_qwen"],
            "Delta PPL": "",
            "Dist-3": metrics["dist-3"],
            "BERTScore": "",
            "Toks/s": "",
        }
        if task == "toxicity":
            row["Avg. Toxicity"] = metrics["avg_toxicity"]
        writer.writerow(row)


def extract(task: str, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    source = TOX_SOURCE if task == "toxicity" else NLI_SOURCE
    source_rows = load_jsonl(source)
    k = len(source_rows[0]["generations"])
    if task == "toxicity":
        toxicity_sidecar = Path(f"{source}-results.txt.toxicity")
        indices = external_toxicity_indices(toxicity_sidecar, generations_per_prompt=k)
        definition = "A = Perspective API TOXICITY summaryScore > 0.5"
        suffixes = ["ppl-big-qwen", "fluency", "repetitions"]
    else:
        indices = external_nli_indices(Path(f"{source}-results.txt.nli"))
        definition = "A = external NLI ensemble class == contradiction"
        suffixes = ["nli", "ppl-big-qwen", "fluency", "repetitions"]
    rows = grouped_subset(source_rows, indices)
    output_dir.mkdir(parents=True)
    output_base = output_dir / "original_external_evaluator_A.jsonl"
    output_base.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    common = write_common_sidecars(output_base, source, indices, suffixes)
    if task == "toxicity":
        toxicity_lines = write_toxicity_sidecar(output_base, toxicity_sidecar, indices, k)
        metrics = toxicity_metrics(rows, toxicity_lines, common)
    else:
        metrics = nli_metrics(rows, common)
    manifest = {
        "task": task,
        "set_definition": definition,
        "source_jsonl": str(source),
        "external_evaluator_sidecar": str(toxicity_sidecar if task == "toxicity" else Path(f"{source}-results.txt.nli")),
        "selected_generation_count": len(indices),
        "selected_prompt_count": len(rows),
        "raveled_indices_file": "external_evaluator_A_indices.txt",
        "paper_metrics_notes": {
            "PPL": "token-aggregated Qwen2.5-14B conditional perplexity from the saved sidecar",
            "BERTScore": "blank: no original-test BERTScore sidecar exists; it was not recomputed",
            "Toks/s": "blank: no comparable original-generation decode-time record was used",
        },
    }
    (output_dir / "external_evaluator_A_indices.txt").write_text(" ".join(map(str, indices)) + "\n", encoding="utf-8")
    write_reports(output_dir, task, metrics, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("toxicity", "nli"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    extract(args.task, args.output_dir)


if __name__ == "__main__":
    main()
