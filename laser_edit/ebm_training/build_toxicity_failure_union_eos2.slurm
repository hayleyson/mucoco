#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=toxicity_union_eos2
#SBATCH --output=/data4/hyeryung/mucoco/slurm_logs/toxicity_union_eos2_%j.out

set -euo pipefail

source /home/hyeryung/.bashrc
source /home/hyeryung/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

cd /data4/hyeryung/mucoco
export PYTHONPATH=.

srun -n 1 -c 1 python - <<'PY'
import json
from pathlib import Path

from laser_edit.ebm_training.build_failure_union_toxicity_dataset import (
    build_union_toxicity_dataset,
)
from laser_edit.ebm_training.failure_union_utils import load_index_file
from laser_edit.utils.utils import read_outputs

THRESHOLD_RECORD = Path(
    "/home/hyeryung/data/loc_edit/models/"
    "roberta-base-jigsaw-toxicity-classifier-energy-training/"
    "step_1000_best_checkpoint/classification_threshold_eos2_20260902.json"
)
F1_THRESHOLD = json.loads(THRESHOLD_RECORD.read_text())["f1_threshold"]
F1_LABEL = f"val_best_f1_{str(F1_THRESHOLD).replace('.', '_')}"

items = {
    "gemini-3.1-pro-preview": Path(
        "laser_edit/base_lm_generate/baselm_gens/gemini-3.1-pro-preview/nontoxic/"
        "gemini-3.1-pro-preview_nontoxic_0shot_system_n250_k10_low_20260824090419.jsonl"
    ),
    "claude-sonnet-4-6": Path(
        "laser_edit/base_lm_generate/baselm_gens/claude-sonnet-4-6/nontoxic/"
        "claude-sonnet-4-6_nontoxic_0shot_system_n250_k10_low_20260824090358.jsonl"
    ),
}

for model_name, jsonl_path in items.items():
    records, summary = build_union_toxicity_dataset(
        model_name, jsonl_path, fixed_threshold=0.95, fixed_threshold_label="0_95"
    )
    for record in records:
        for generation in record["generations"]:
            flags = generation["failure_source_flags"]
            if flags.pop("energy_fixed_0_95"):
                flags["energy_fixed_0_95_eos2"] = True
                generation["failure_sources"] = [
                    "energy_fixed_0_95_eos2"
                    if source == "energy_fixed_0_95"
                    else source
                    for source in generation["failure_sources"]
                ]
            else:
                flags["energy_fixed_0_95_eos2"] = False
            flags["energy_val_best_f1_eos2"] = False

    base = jsonl_path.name.removesuffix(".jsonl")
    f1_index_path = jsonl_path.with_name(
        f"{base}_edit_candidates_{F1_LABEL}_index.txt"
    )
    f1_indices = load_index_file(f1_index_path)
    flat = read_outputs(str(jsonl_path))
    generations_per_prompt = summary["generations_per_prompt"]

    by_premise = {}
    by_index = {}
    for record in records:
        for generation in record["generations"]:
            premise_index = generation["raveled_index"] // generations_per_prompt
            by_premise[premise_index] = record
            by_index[generation["raveled_index"]] = generation

    for index in f1_indices:
        if index in by_index:
            generation = by_index[index]
            generation["failure_source_flags"]["energy_val_best_f1_eos2"] = True
            generation["failure_sources"].append("energy_val_best_f1_eos2")
            continue

        premise_index = index // generations_per_prompt
        if premise_index not in by_premise:
            row = flat.iloc[index]
            by_premise[premise_index] = {
                "prompt": {"text": row["prompt"]},
                "generations": [],
            }
        generation = {
            "text": flat.iloc[index]["text"],
            "raveled_index": index,
            "generation_index": index % generations_per_prompt,
            "failure_sources": ["energy_val_best_f1_eos2"],
            "failure_source_flags": {
                "perspective_api_toxic": False,
                "energy_fixed_0_95_eos2": False,
                "energy_val_best_f1_eos2": True,
            },
        }
        by_premise[premise_index]["generations"].append(generation)
        by_index[index] = generation

    records = [by_premise[index] for index in sorted(by_premise)]
    for record in records:
        record["generations"].sort(key=lambda generation: generation["generation_index"])

    source_b_size = summary["subset_sizes"].pop("energy_fixed_0_95")
    summary["subset_sizes"]["energy_fixed_0_95_eos2"] = source_b_size
    summary["thresholds"]["energy_fixed_0_95_eos2"] = summary["thresholds"].pop(
        "energy_fixed_0_95"
    )
    summary["thresholds"]["energy_val_best_f1_eos2"] = F1_THRESHOLD
    summary["subset_sizes"]["energy_val_best_f1_eos2"] = len(f1_indices)
    summary["subset_sizes"]["union"] = len(by_index)
    summary["output_generation_count"] = len(by_index)
    summary["output_prompt_count"] = len(records)
    summary["energy_scoring"] = {
        "terminal_eos_count": 2,
        "fixed_0_95_index_path": str(jsonl_path.with_name(f"{base}_edit_candidates_0_95_index.txt")),
        "validation_best_f1_index_path": str(f1_index_path),
        "threshold_record": str(THRESHOLD_RECORD),
    }

    out_path = jsonl_path.with_name(f"{base}_failure_union_abc_eos2_20260902.jsonl")
    summary_path = jsonl_path.with_name(
        f"{base}_failure_union_abc_eos2_20260902.summary.json"
    )
    with out_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{model_name}: wrote {summary['output_prompt_count']} prompts "
        f"({summary['output_generation_count']} generations) to {out_path}"
    )
PY
