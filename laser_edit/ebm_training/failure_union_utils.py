#!/usr/bin/env python3
"""Inspect intersections among three not-satisfied subsets per model."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/hyeryung/data/mucoco")
MODEL_PATH = Path(
    "/home/hyeryung/data/loc_edit/models/nli/"
    "roberta_large_snli_mnli_anli_train_dev_with_finegrained_finegrained_labels_cross_entropy_n_a/zgs9e2sr/"
)
VALIDATION_DATASET = ROOT / "laser_edit/ebm_training/data/nli/snli_mnli_anli_train_dev_with_finegrained.jsonl"

FILES = {
    "gemini-3.1-pro-preview": ROOT
    / "laser_edit/base_lm_generate/baselm_gens/gemini-3.1-pro-preview/anli-r2-test/"
    "gemini-3.1-pro-preview_nli_0shot_n1000_k10_low_20260819003604.jsonl",
    "claude-sonnet-4-6": ROOT
    / "laser_edit/base_lm_generate/baselm_gens/claude-sonnet-4-6/anli-r2-test/"
    "claude-sonnet-4-6_nli_0shot_n1000_k10_low_20260819003304.jsonl",
}


def load_index_file(path: Path) -> set[int]:
    text = path.read_text().strip()
    if not text:
        return set()
    return {int(x) for x in text.split()}


def load_nli_contradiction_indices(nli_path: Path) -> set[int]:
    indices: set[int] = set()
    with nli_path.open() as f:
        for i, line in enumerate(f):
            row = ast.literal_eval(line.strip())
            if row.get("nli_class") == "contradiction":
                indices.add(i)
    return indices


def ensure_val_best_f1_indices(jsonl_path: Path, f1_threshold: float, label: str) -> Path:
    from laser_edit.ebm_training.find_classification_threshold import (
        save_testset_edit_candidates,
    )

    index_path = jsonl_path.with_name(
        f"{jsonl_path.name.rsplit('.jsonl', 1)[0]}_edit_candidates_{label}_index.txt"
    )
    if index_path.exists():
        return index_path

    save_testset_edit_candidates(
        str(MODEL_PATH),
        str(jsonl_path),
        "nli",
        label_id=1,
        threshold=f1_threshold,
        batch_size=64,
        num_workers=2,
        threshold_label=label,
        threshold_source="validation_best_f1",
    )
    return index_path


def subset_report(name: str, universe: int, a: set[int], b: set[int], c: set[int]) -> dict:
    only_a = a - b - c
    only_b = b - a - c
    only_c = c - a - b
    ab_only = (a & b) - c
    ac_only = (a & c) - b
    bc_only = (b & c) - a
    abc = a & b & c
    full = set(range(universe))
    union = a | b | c
    none = len(full - union)

    def pct(n: int) -> float:
        return 100.0 * n / universe if universe else 0.0

    report = {
        "model": name,
        "universe": universe,
        "sizes": {
            "A_external_nli_contradiction": len(a),
            "B_energy_fixed_0_99": len(b),
            "C_energy_val_best_f1": len(c),
        },
        "pairwise_intersections": {
            "A_and_B": len(a & b),
            "A_and_C": len(a & c),
            "B_and_C": len(b & c),
            "A_and_B_and_C": len(abc),
        },
        "exclusive_regions": {
            "only_A": len(only_a),
            "only_B": len(only_b),
            "only_C": len(only_c),
            "A_and_B_not_C": len(ab_only),
            "A_and_C_not_B": len(ac_only),
            "B_and_C_not_A": len(bc_only),
            "all_three": len(abc),
            "none_of_three": none,
        },
        "percent_of_universe": {
            k: round(pct(v), 3)
            for k, v in {
                "A": len(a),
                "B": len(b),
                "C": len(c),
                "A_and_B": len(a & b),
                "A_and_C": len(a & c),
                "B_and_C": len(b & c),
                "A_and_B_and_C": len(abc),
                "only_A": len(only_a),
                "only_B": len(only_b),
                "only_C": len(only_c),
                "none_of_three": none,
            }.items()
        },
    }
    return report


def format_report(report: dict, f1_threshold: float) -> str:
    lines = [
        f"Model: {report['model']}",
        f"Universe: {report['universe']} raveled generations",
        "",
        "Subset definitions:",
        "  A = external NLI evaluator contradiction (nli_class == 'contradiction')",
        "  B = energy not-satisfied at fixed threshold 0.99 (very conservative)",
        f"  C = energy not-satisfied at validation best-F1 threshold {f1_threshold:.6f}",
        "",
        "Subset sizes:",
        f"  |A| = {report['sizes']['A_external_nli_contradiction']} ({report['percent_of_universe']['A']:.3f}%)",
        f"  |B| = {report['sizes']['B_energy_fixed_0_99']} ({report['percent_of_universe']['B']:.3f}%)",
        f"  |C| = {report['sizes']['C_energy_val_best_f1']} ({report['percent_of_universe']['C']:.3f}%)",
        "",
        "Pairwise / triple intersections:",
        f"  |A ∩ B| = {report['pairwise_intersections']['A_and_B']} ({report['percent_of_universe']['A_and_B']:.3f}%)",
        f"  |A ∩ C| = {report['pairwise_intersections']['A_and_C']} ({report['percent_of_universe']['A_and_C']:.3f}%)",
        f"  |B ∩ C| = {report['pairwise_intersections']['B_and_C']} ({report['percent_of_universe']['B_and_C']:.3f}%)",
        f"  |A ∩ B ∩ C| = {report['pairwise_intersections']['A_and_B_and_C']} ({report['percent_of_universe']['A_and_B_and_C']:.3f}%)",
        "",
        "Venn exclusive regions:",
        f"  only A = {report['exclusive_regions']['only_A']} ({report['percent_of_universe']['only_A']:.3f}%)",
        f"  only B = {report['exclusive_regions']['only_B']} ({report['percent_of_universe']['only_B']:.3f}%)",
        f"  only C = {report['exclusive_regions']['only_C']} ({report['percent_of_universe']['only_C']:.3f}%)",
        f"  A∩B only = {report['exclusive_regions']['A_and_B_not_C']}",
        f"  A∩C only = {report['exclusive_regions']['A_and_C_not_B']}",
        f"  B∩C only = {report['exclusive_regions']['B_and_C_not_A']}",
        f"  all three = {report['exclusive_regions']['all_three']}",
        f"  none of A/B/C = {report['exclusive_regions']['none_of_three']} ({report['percent_of_universe']['none_of_three']:.3f}%)",
        "",
        "Complements within universe:",
        f"  A complement = {report['universe'] - report['sizes']['A_external_nli_contradiction']}",
        f"  B complement = {report['universe'] - report['sizes']['B_energy_fixed_0_99']}",
        f"  C complement = {report['universe'] - report['sizes']['C_energy_val_best_f1']}",
        f"  (A ∪ B ∪ C) complement = {report['exclusive_regions']['none_of_three']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    threshold_json = json.loads((MODEL_PATH / "classification_threshold.json").read_text())
    f1_threshold = float(threshold_json["f1_threshold"])
    f1_label = f"val_best_f1_{str(f1_threshold).replace('.', '_')}"

    all_reports = []
    text_blocks = [
        "Three-subset intersection analysis",
        f"Validation best-F1 threshold: {f1_threshold}",
        "",
    ]

    for model_name, jsonl_path in FILES.items():
        base = jsonl_path.name.rsplit(".jsonl", 1)[0]
        parent = jsonl_path.parent

        nli_path = jsonl_path.with_name(jsonl_path.name + "-results.txt.nli")
        b_index_path = parent / f"{base}_edit_candidates_0_99_index.txt"
        c_index_path = ensure_val_best_f1_indices(jsonl_path, f1_threshold, f1_label)

        a = load_nli_contradiction_indices(nli_path)
        b = load_index_file(b_index_path)
        c = load_index_file(c_index_path)

        universe = 10000
        if len(a | b | c) > universe:
            universe = max(max(a, default=-1), max(b, default=-1), max(c, default=-1)) + 1

        report = subset_report(model_name, universe, a, b, c)
        all_reports.append(report)
        text_blocks.append(format_report(report, f1_threshold))

    out_json = ROOT / "laser_edit/ebm_training/subset_intersection_1000x10_low.json"
    out_txt = ROOT / "laser_edit/ebm_training/subset_intersection_1000x10_low.txt"
    out_json.write_text(json.dumps(all_reports, indent=2) + "\n")
    out_txt.write_text("\n".join(text_blocks))
    print(out_txt.read_text())


if __name__ == "__main__":
    main()
