import json
from pathlib import Path

from laser_edit.ebm_training.build_failure_union_toxicity_dataset import (
    load_perspective_toxic_indices,
)


def test_failure_union_a_indices_use_summary_score(tmp_path: Path):
    sidecar = tmp_path / "source.jsonl-results.txt.toxicity"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps({"allresponses": {
                    "gen-0-0-0": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.4}}], "summaryScore": {"value": 0.6}}}}, None],
                    "gen-0-0-1": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.6}}], "summaryScore": {"value": 0.4}}}}, None],
                }}),
                json.dumps({"allresponses": {
                    "gen-1-0-0": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.7}}], "summaryScore": {"value": 0.2}}}}, None],
                    "gen-1-0-1": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.2}}], "summaryScore": {"value": 0.9}}}}, None],
                }}),
            ]
        ) + "\n",
        encoding="utf-8",
    )

    assert load_perspective_toxic_indices(sidecar, generations_per_prompt=2) == {0, 3}
