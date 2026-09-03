#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=gpt35_tox_union
#SBATCH --output=/data4/hyeryung/mucoco/slurm_logs/gpt35_tox_union_%j.out

set -euo pipefail

source /home/hyeryung/.bashrc
source /home/hyeryung/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

cd /data4/hyeryung/mucoco
export PYTHONPATH=.
export HF_HOME=/home/hyeryung/data/hf_cache

srun -n 1 -c 1 python - <<'PY'
import json
from pathlib import Path

from laser_edit.ebm_training.build_failure_union_dataset import group_records_for_edit_input
from laser_edit.ebm_training.build_failure_union_toxicity_dataset import (
    infer_generations_per_prompt,
    load_perspective_toxic_indices,
)
from laser_edit.ebm_training.find_classification_threshold import (
    find_classification_threshold,
    save_testset_edit_candidates,
)
from laser_edit.ebm_training.failure_union_utils import load_index_file
from laser_edit.utils.utils import read_outputs

model_path = "/home/hyeryung/data/loc_edit/models/roberta-base-jigsaw-toxicity-classifier-energy-training/step_1000_best_checkpoint"
validation_path = (
    "/data4/hyeryung/mucoco/laser_edit/ebm_training/data/toxicity/"
    "jigsaw-unintended-bias-in-toxicity-classification/fine-grained/valid.jsonl"
)
jsonl_path = Path(
    "/home/hyeryung/data/mucoco/laser_edit/base_lm_generate/baselm_gens/"
    "gpt-3.5-turbo-0125/nontoxic/"
    "gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150.jsonl"
)
fixed_index_path = jsonl_path.with_name(
    "gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_"
    "below_nontoxic_threshold_0_95_332_index.txt"
)

# C: recompute calibration and save the two-EOS best-F1 subset with the maintained filter.
threshold_result = find_classification_threshold(
    model_path,
    validation_path,
    task="toxicity",
    label_id=0,
    batch_size=64,
    num_workers=0,
)
threshold_path = jsonl_path.with_name(
    jsonl_path.name.removesuffix(".jsonl")
    + "_classification_threshold_eos2_20260902.json"
)
threshold_path.write_text(json.dumps(threshold_result, indent=2) + "\n", encoding="utf-8")
f1_threshold = threshold_result["f1_threshold"]
f1_label = f"val_best_f1_{str(f1_threshold).replace('.', '_')}"
_, f1_index_path = save_testset_edit_candidates(
    model_path,
    str(jsonl_path),
    task="toxicity",
    label_id=0,
    threshold=f1_threshold,
    batch_size=64,
    num_workers=0,
    threshold_label=f1_label,
    threshold_source="validation_best_f1_eos2_20260902",
)
f1_index_path = Path(f1_index_path)

# A: external Perspective toxicity failure; B: supplied historical fixed-0.95 index;
# C: newly calibrated two-EOS validation-best-F1 index.
generations_per_prompt = infer_generations_per_prompt(jsonl_path)
flat = read_outputs(str(jsonl_path))
a = load_perspective_toxic_indices(
    jsonl_path.with_name(jsonl_path.name + "-results.txt.toxicity"),
    generations_per_prompt=generations_per_prompt,
    toxic_threshold=0.5,
)
b = load_index_file(fixed_index_path)
c = load_index_file(f1_index_path)
union = sorted(a | b | c)

records = []
for index in union:
    row = flat.iloc[index]
    flags = {
        "perspective_api_toxic": index in a,
        "energy_fixed_0_95_eos2": index in b,
        "energy_val_best_f1_eos2": index in c,
    }
    sources = [name for name, present in flags.items() if present]
    records.append(
        {
            "raveled_index": index,
            "premise_index": index // generations_per_prompt,
            "generation_index": index % generations_per_prompt,
            "prompt": {"text": row["prompt"]},
            "generation": {"text": row["text"]},
            "failure_sources": sources,
            "failure_source_flags": flags,
        }
    )

grouped = group_records_for_edit_input(records)
base = jsonl_path.name.removesuffix(".jsonl")
out_path = jsonl_path.with_name(f"{base}_failure_union_abc_eos2_20260902.jsonl")
summary_path = jsonl_path.with_name(f"{base}_failure_union_abc_eos2_20260902.summary.json")
with out_path.open("w", encoding="utf-8") as output_file:
    for record in grouped:
        output_file.write(json.dumps(record) + "\n")

summary = {
    "model": "gpt-3.5-turbo-0125",
    "source_jsonl": str(jsonl_path),
    "perspective_sidecar": str(jsonl_path.with_name(jsonl_path.name + "-results.txt.toxicity")),
    "universe_size": len(flat),
    "generations_per_prompt": generations_per_prompt,
    "prompt_count": len(flat) // generations_per_prompt,
    "thresholds": {
        "perspective_api_toxic": 0.5,
        "energy_fixed_0_95_eos2": 0.95,
        "energy_val_best_f1_eos2": f1_threshold,
    },
    "subset_sizes": {
        "perspective_api_toxic": len(a),
        "energy_fixed_0_95_eos2": len(b),
        "energy_val_best_f1_eos2": len(c),
        "union": len(union),
    },
    "output_generation_count": len(records),
    "output_prompt_count": len(grouped),
    "output_format": "prompt_generations",
    "energy_scoring": {
        "terminal_eos_count": 2,
        "fixed_0_95_index_path": str(fixed_index_path),
        "validation_best_f1_index_path": str(f1_index_path),
        "threshold_record": str(threshold_path),
    },
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Wrote {out_path}")
print(f"Wrote {summary_path}")
PY
