"""
Unit tests for set_consistency_llm_edit (prompt building + parsing + path helpers).

These tests mock OpenAI / vLLM so they run without API keys or GPUs.
Run from repo root with:
  cd /path/to/mucoco && python -m pytest new_module/test/test_set_consistency_llm_edit.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _chdir_repo():
    old = os.getcwd()
    old_path = sys.path[:]
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "new_module"))
    try:
        yield
    finally:
        os.chdir(old)
        sys.path[:] = old_path
        sys.modules.pop("set_consistency_llm_edit", None)


@pytest.fixture(scope="module")
def sce():
    import sys
    from unittest.mock import MagicMock

    if "vllm" not in sys.modules:
        _vllm = MagicMock()
        _vllm.LLM = MagicMock
        _vllm.SamplingParams = MagicMock
        sys.modules["vllm"] = _vllm

    import set_consistency_llm_edit as m

    return m


def test_pkl_path_joins_expected_components(sce):
    p = sce._pkl_path("lconvqa", "test", "C")
    # Implementation uses a repo-relative path (cwd is repo root in main / loader usage).
    assert p == Path("new_module/data/convqa/lconvqa_test_C_dataset.pickle")


def test_gpt_set_prompt_wo_locate_lconvqa(sce):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
        with patch.object(sce, "OpenAI", return_value=MagicMock()):
            g = sce.GPT("gpt-4o-mini", None, "lconvqa")
    data = ["question: q1, answer: a1.", "question: q2, answer: a2."]
    prompt = g.set_prompt(data, located_indexes=None)
    assert "(1) question: q1, answer: a1." in prompt
    assert "question-answer pair" in prompt
    assert "Indexes to Edit" not in prompt


def test_gpt_set_prompt_with_locate_formats_list(sce):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
        with patch.object(sce, "OpenAI", return_value=MagicMock()):
            g = sce.GPT("gpt-4o-mini", None, "set_nli")
    data = ["First sentence.", "Second sentence."]
    prompt = g.set_prompt(data, located_indexes=[1, 3])
    assert "sentence indexes to edit:" in prompt.lower()
    assert "- [1, 3]" in prompt


def test_parse_pairs_response_splits_numbered_entries(sce):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
        with patch.object(sce, "OpenAI", return_value=MagicMock()):
            g = sce.GPT("gpt-4o-mini", None, "lconvqa")
    text = "(1) question: Q?, answer: A. (2) question: Q2?, answer: A2"
    pairs = g.parse_pairs_response(text)
    assert pairs[0][0] == "Q?"
    assert pairs[0][1] == "A"
    assert pairs[1][0] == "Q2?"
    assert pairs[1][1] == "A2"


def test_parse_sentences_response(sce):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
        with patch.object(sce, "OpenAI", return_value=MagicMock()):
            g = sce.GPT("gpt-4o-mini", None, "set_nli")
    text = "(1) Hello world. (2) Second line."
    sents = g.parse_sentences_response(text)
    assert sents == ["Hello world.", "Second line."]


def test_hf_parse_strips_redacted_thinking_suffix(sce):
    """HFModel parsing should ignore content before closing thinking tag."""
    mock_llm = MagicMock()
    mock_tok = MagicMock()
    mock_tok.apply_chat_template = lambda *a, **k: ""
    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained.return_value = mock_tok
    with patch.object(sce, "LLM", return_value=mock_llm), patch.object(
        sce, "AutoTokenizer", mock_tok_cls
    ):
        m = sce.HFModel("dummy-model", "set_nli", tensor_parallel_size=1)
    raw = "<think>...</think>\n(1) One. (2) Two."
    assert m.parse_sentences_response(raw) == ["One.", "Two."]


def test_hf_parse_truncated_thinking_returns_empty(sce):
    mock_llm = MagicMock()
    mock_tok = MagicMock()
    mock_tok.apply_chat_template = lambda *a, **k: ""
    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained.return_value = mock_tok
    with patch.object(sce, "LLM", return_value=mock_llm), patch.object(
        sce, "AutoTokenizer", mock_tok_cls
    ):
        m = sce.HFModel("dummy-model", "lconvqa", tensor_parallel_size=1)
    raw = "<think>incomplete"
    assert m.parse_pairs_response(raw) == []
    assert m.parse_sentences_response(raw) == []


def test_unknown_dataset_name_skips_init_fields(sce):
    """Unsupported dataset_name leaves datapoint_type / parse_response unset (bug pitfall)."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
        with patch.object(sce, "OpenAI", return_value=MagicMock()):
            g = sce.GPT("gpt-4o-mini", None, "unknown_dataset")
    assert not hasattr(g, "datapoint_type")
