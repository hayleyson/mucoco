import json
from pathlib import Path

from laser_edit.evaluation.extract_external_evaluator_failure_subsets import (
    external_toxicity_indices,
    extract_lines,
    write_reports,
)


def test_extracts_perspective_toxic_generation_indices_and_sidecar_lines(tmp_path: Path):
    toxicity = tmp_path / "source.jsonl-results.txt.toxicity"
    toxicity.write_text(
        "\n".join(
            [
                json.dumps({"allresponses": {
                    "gen-0-9-0": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.4}}], "summaryScore": {"value": 0.6}}}}, None],
                    "gen-0-9-1": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.6}}], "summaryScore": {"value": 0.4}}}}, None],
                }}),
                json.dumps({"allresponses": {
                    "gen-1-9-0": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.7}}], "summaryScore": {"value": 0.2}}}}, None],
                    "gen-1-9-1": [{"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.2}}], "summaryScore": {"value": 0.9}}}}, None],
                }}),
            ]
        ) + "\n",
        encoding="utf-8",
    )

    assert external_toxicity_indices(toxicity, generations_per_prompt=2) == [0, 3]
    assert extract_lines(["zero", "one", "two", "three"], [0, 3]) == ["zero", "three"]


def test_toxicity_paper_csv_includes_average_toxicity_column(tmp_path: Path):
    metrics = {
        "constraint_satisfaction": 0.0,
        "avg_toxicity": 0.6,
        "total_ppl_qwen": 4.9,
        "dist-3": 0.9,
    }
    write_reports(tmp_path, "toxicity", metrics, {})

    header = (tmp_path / "paper_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "Ctrl. Acc.,Avg. Toxicity,PPL,Delta PPL,Dist-3,BERTScore,Toks/s"
