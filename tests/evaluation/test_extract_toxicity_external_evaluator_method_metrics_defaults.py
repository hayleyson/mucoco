import sys
from pathlib import Path
from unittest.mock import patch

import laser_edit.evaluation.extract_toxicity_external_evaluator_method_metrics as module


def test_method_metrics_defaults_select_and_score_with_summary_score(tmp_path: Path):
    captured = {}

    def capture_indices(score_kind):
        captured["selection_score_kind"] = score_kind
        return []

    def capture_extract_method(*args, **kwargs):
        captured["metric_score_kind"] = kwargs["metric_score_kind"]
        captured["extract_selection_score_kind"] = kwargs["selection_score_kind"]

    with (
        patch.object(module, "METHODS", {"method": ("Method", tmp_path / "outputs.jsonl", "full")}),
        patch.object(module, "load_jsonl", return_value=[]),
        patch.object(module, "toxicity_indices", side_effect=capture_indices),
        patch.object(module, "ppl_metrics", return_value=(0.0, 0.0)),
        patch.object(module, "extract_method", side_effect=capture_extract_method),
        patch.object(sys, "argv", ["metrics", "--output-root", str(tmp_path / "report")]),
    ):
        module.main()

    assert captured == {
        "selection_score_kind": "summary_score",
        "metric_score_kind": "summary_score",
        "extract_selection_score_kind": "summary_score",
    }
