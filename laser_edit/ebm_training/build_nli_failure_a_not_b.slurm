#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --job-name=nli_a_not_b
#SBATCH --output=/home/hyeryung/data/mucoco/laser_edit/ebm_training/build_nli_failure_a_not_b_%j.out

set -euo pipefail

source /home/${USER}/.bashrc
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

ROOT="/home/hyeryung/data/mucoco"
cd "$ROOT"
export PYTHONPATH=.

srun -n 1 -c 1 python - <<'PY'
import json
from pathlib import Path

from laser_edit.ebm_training.build_failure_union_dataset import (
    SOURCE_A,
    SOURCE_B,
    group_records_for_edit_input,
    infer_generations_per_prompt,
    load_nli_rows,
)
from laser_edit.ebm_training.failure_union_utils import (
    load_index_file,
    load_nli_contradiction_indices,
)
from laser_edit.utils.utils import read_outputs

source_path = Path(
    "/home/hyeryung/data/mucoco/laser_edit/data/logical-consistency/"
    "anli-r2-test_prompt_4.jsonl"
)
base = source_path.name.rsplit(".jsonl", 1)[0]
nli_path = source_path.with_name(source_path.name + "-results.txt.nli")
b_index_path = source_path.with_name(f"{base}_edit_candidates_0_99_index.txt")

a = load_nli_contradiction_indices(nli_path)
b = load_index_file(b_index_path)
selected = sorted(a - b)
flat = read_outputs(str(source_path))
nli_rows = load_nli_rows(nli_path)
generations_per_prompt = infer_generations_per_prompt(source_path)
if len(flat) != len(nli_rows):
    raise RuntimeError(f"Row count mismatch: generations={len(flat)} nli={len(nli_rows)}")

records = []
for idx in selected:
    row = flat.iloc[idx]
    nli = nli_rows[idx]
    records.append({
        "raveled_index": idx,
        "premise_index": idx // generations_per_prompt,
        "generation_index": idx % generations_per_prompt,
        "prompt": {"text": row["prompt"]},
        "generation": {"text": row["text"]},
        "failure_sources": [SOURCE_A],
        "failure_source_flags": {SOURCE_A: True, SOURCE_B: False},
        "external_nli": {
            "nli_class": nli.get("nli_class"),
            "entailment_prob": nli.get("entailment_prob"),
            "neutral_prob": nli.get("neutral_prob"),
            "contradiction_prob": nli.get("contradiction_prob"),
        },
    })

grouped = group_records_for_edit_input(records)
out_path = source_path.with_name(f"{base}_failure_a_not_b_20260902.jsonl")
index_path = source_path.with_name(f"{base}_failure_a_not_b_20260902_index.txt")
summary_path = source_path.with_name(f"{base}_failure_a_not_b_20260902.summary.json")
with out_path.open("w") as f:
    for record in grouped:
        f.write(json.dumps(record) + "\n")
index_path.write_text(" ".join(str(idx) for idx in selected) + "\n")
summary_path.write_text(json.dumps({
    "source_jsonl": str(source_path),
    "nli_sidecar": str(nli_path),
    "fixed_energy_index": str(b_index_path),
    "set_definition": "A minus B; A=external NLI contradiction, B=energy fixed threshold 0.99",
    "output_generation_count": len(selected),
    "output_prompt_count": len(grouped),
    "output_format": "prompt_generations",
}, indent=2) + "\n")
print(f"A={len(a)} B={len(b)} A_not_B={len(selected)}")
print(f"Wrote {out_path}")
print(f"Wrote {index_path}")
print(f"Wrote {summary_path}")
PY
