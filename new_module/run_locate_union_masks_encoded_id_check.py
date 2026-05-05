#!/usr/bin/env python
# coding: utf-8
"""
Run only multi-locator `call_locate` + `union_masks` paths from new_mlm_reranking_all.

Detects:
  - Length mismatch: locator outputs tokenize under the MLM tokenizer to different lengths
    (raises ValueError inside union_masks).
  - Token-ID mismatch at aligned positions where *no* variant has the MLM mask token —
    merging with stacked[0] would silently ignore other variants at those indices.

Requires len(losses) > 2 so the union_masks branch matches the reranking pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

import new_module.losses as lossbuilder
from new_module.ebm_training.nli.models import EncoderModel
from new_module.locate.new_locate_utils import LocateMachine
from new_module.new_mlm_reranking_all import call_locate, union_masks
from new_module.utils.robertacustom import RobertaCustomForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("LOGGING_LEVEL", "INFO")


class DummyArgs:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def diagnose_union_mask_encodings(
    masked_texts_per_loss: List[List[str]],
    mlm_tokenizer: AutoTokenizer,
) -> List[Dict[str, Any]]:
    """
    Same encoding contract as union_masks (add_special_tokens=True).

    Reports:
      - length_mismatch: sequence lengths differ across locator strings for row b.
      - token_id_mismatch: aligned position j where no row has MASK but token ids disagree.
    """
    issues: List[Dict[str, Any]] = []
    if not masked_texts_per_loss:
        return issues

    batch_size = len(masked_texts_per_loss[0])
    mask_id = mlm_tokenizer.mask_token_id

    for b in range(batch_size):
        variants = [loss_texts[b] for loss_texts in masked_texts_per_loss]
        encoded_tensors = [
            mlm_tokenizer.encode(t, add_special_tokens=True, return_tensors="pt")
            for t in variants
        ]
        seq_lens = {t.shape[1] for t in encoded_tensors}
        if len(seq_lens) != 1:
            issues.append(
                {
                    "batch_row": b,
                    "kind": "length_mismatch",
                    "seq_lens": sorted(seq_lens),
                    "variant_previews": [v[:160] + ("..." if len(v) > 160 else "") for v in variants],
                }
            )
            continue

        stacked = torch.cat(encoded_tensors, dim=0)  # [num_variants, seq_len]
        seq_len = stacked.shape[1]
        for j in range(seq_len):
            col = stacked[:, j].tolist()
            if mask_id is not None and mask_id in col:
                continue
            if len(set(col)) > 1:
                issues.append(
                    {
                        "batch_row": b,
                        "kind": "token_id_mismatch",
                        "position": j,
                        "token_ids_per_variant": col,
                        "tokens_decoded_per_variant": [
                            mlm_tokenizer.decode([tid]) for tid in col
                        ],
                    }
                )

    return issues


def setup_losses_and_locators(config: Dict[str, Any]):
    """Mirror new_mlm_reranking_all.main model / loss / locator setup (no MLM LM weights)."""
    name2tokenizer: Dict[str, Any] = {}
    name2model: Dict[str, Any] = {}
    name2config: Dict[str, Any] = {}

    for i, model_path in enumerate(config["model_paths"]):
        if model_path in name2model:
            continue

        if config["model_types"][i] == "EncoderModel":
            with open(os.path.join(config["model_paths"][i], "config.json")) as f:
                model_config = json.load(f)
            model_config["device"] = config["device"]
            model_config["model_path"] = os.path.join(
                config["model_paths"][i], "best_model_pearsonr.pth"
            )
            if config["locate_method"] == "attention":
                model_config["locate"]["type"] = "attention"
            elif config["locate_method"] == "grad_norm":
                model_config["locate"]["type"] = "gradnorm"
            name2config[model_path] = model_config

            model = EncoderModel(params=name2config[model_path])
            model.load_state_dict(
                torch.load(name2config[model_path]["model_path"], weights_only=True),
                strict=False,
            )
            name2model[model_path] = lossbuilder.ModelWrapper(model)
            name2model[model_path].eval()
            name2model[model_path].to(config["device"])

            name2tokenizer[config["tokenizer_paths"][i]] = name2model[model_path].tokenizer

        else:
            name2config[model_path] = AutoConfig.from_pretrained(
                model_path, cache_dir=config["cache_dir"]
            )

            if config["model_types"][i] == "RobertaCustomForSequenceClassification":
                name2model[model_path] = lossbuilder.ModelWrapper(
                    RobertaCustomForSequenceClassification.from_pretrained(
                        model_path,
                        config=name2config[model_path],
                        cache_dir=config["cache_dir"],
                    )
                )
            else:
                name2model[model_path] = lossbuilder.ModelWrapper(
                    getattr(transformers, config["model_types"][i]).from_pretrained(
                        model_path,
                        config=name2config[model_path],
                        cache_dir=config["cache_dir"],
                        use_safetensors=True,
                    )
                )
            name2model[model_path].eval()
            name2model[model_path].to(config["device"])

            try:
                name2tokenizer[config["tokenizer_paths"][i]] = AutoTokenizer.from_pretrained(
                    config["tokenizer_paths"][i],
                    cache_dir=config["cache_dir"],
                    use_fast=True,
                )
            except Exception:
                name2tokenizer[config["tokenizer_paths"][i]] = AutoTokenizer.from_pretrained(
                    config["tokenizer_paths"][i],
                    cache_dir=config["cache_dir"],
                    use_fast=False,
                )

    name2model[config["model_paths"][0]].half()

    mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    build_loss_args = DummyArgs(**config["build_loss_dict"])
    build_loss_args.task = config["task"]

    lossfns: List[Any] = []
    for i, loss in enumerate(config["losses"]):
        lossfns.append(
            lossbuilder.build_loss(
                loss,
                name2model[config["model_paths"][i]],
                name2tokenizer[config["tokenizer_paths"][i]],
                build_loss_args,
            )
        )
        lossfns[i].tokenizer.add_special_tokens({"mask_token": mlm_tokenizer.mask_token})

    if len(config["losses"]) == 2:
        locate_modes = [config["task"]]
    else:
        locate_modes = config["task"].split("_")

    locators = [
        LocateMachine(lossfns[i].model, lossfns[i].tokenizer, locate_modes[i - 1])
        for i in range(1, len(config["losses"]))
    ]

    return lossfns, locators, locate_modes, mlm_tokenizer


def load_generation_rows(
    source_path: str,
    primary_key: str,
    secondary_key: str,
    task: str,
):
    """Match new_mlm_reranking_all dataset layout for toxicity/sentiment/nli."""
    if task == "formality":
        raise ValueError("Formality plain-line format not supported here; use nli/toxicity/sentiment-style JSONL.")
    sources = []
    generations = []
    with open(source_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sources.append(row[primary_key][secondary_key])
            generations.append(row["generations"])
    return sources, generations


def run_check(config: Dict[str, Any]) -> int:
    if len(config["losses"]) <= 2:
        logger.error(
            "This script exercises union_masks only when len(losses) > 2 "
            "(one LM + multiple auxiliary constraints). Got len(losses)=%s.",
            len(config["losses"]),
        )
        return 2

    _lossfns, locators, locate_modes, mlm_tokenizer = setup_losses_and_locators(config)

    if len(locate_modes) != len(locators):
        logger.error(
            "`task` must yield one locate mode per auxiliary loss. "
            "Got len(locators)=%s (%s auxiliary heads) but locate_modes=%s from task=%r.",
            len(locators),
            len(config["losses"]) - 1,
            locate_modes,
            config["task"],
        )
        logger.error(
            "Fix: use an underscore-separated --task "
            '(e.g. "toxicity_sentiment") with exactly %s segments matching each auxiliary model.',
            len(locators),
        )
        return 2

    locate_cfg_keys = {"device", "locate_method", "num_edit_token_per_step", "locate_unit"}
    locate_subcfg = {k: config[k] for k in locate_cfg_keys}

    sources, gens = load_generation_rows(
        config["source_data"],
        config["jsonl_primary_key"],
        config["jsonl_secondary_key"],
        config["task"],
    )

    total_rows = len(sources)
    end = (
        total_rows
        if config["max_prompts"] is None
        else min(total_rows, config["start_prompt"] + config["max_prompts"])
    )

    union_errors = 0
    diag_length = 0
    diag_token_mismatch = 0
    examples: List[Dict[str, Any]] = []

    for text_id in range(config["start_prompt"], end):
        source_text = sources[text_id]
        hypotheses = [x["text"] for x in gens[text_id]]
        if not hypotheses:
            continue

        masked_texts: List[List[str]] = []
        for i in range(len(config["losses"]) - 1):
            masked_texts.append(
                call_locate(
                    locate_modes[i],
                    config["target_label_ids"][i + 1],
                    locators[i],
                    source_text,
                    hypotheses,
                    locate_subcfg,
                )
            )

        diag_issues = diagnose_union_mask_encodings(masked_texts, mlm_tokenizer)

        union_exc: Optional[str] = None
        try:
            _merged = union_masks(masked_texts, mlm_tokenizer)
            assert len(_merged) == len(hypotheses)
        except ValueError as e:
            union_exc = str(e)

        row_report: Dict[str, Any] = {
            "text_id": text_id,
            "num_hypotheses": len(hypotheses),
            "diagnose_issues": diag_issues,
            "union_masks_value_error": union_exc,
        }

        had_len = any(x["kind"] == "length_mismatch" for x in diag_issues)
        had_tid = any(x["kind"] == "token_id_mismatch" for x in diag_issues)
        if had_len:
            diag_length += 1
        if had_tid:
            diag_token_mismatch += 1
        if union_exc is not None:
            union_errors += 1

        if diag_issues or union_exc is not None:
            examples.append(row_report)

        if (text_id - config["start_prompt"] + 1) % config["log_every"] == 0:
            logger.info(
                "Processed prompts %s..%s (union ValueError rows=%s, "
                "rows with length_mismatch diag=%s, rows with token_id_mismatch diag=%s)",
                config["start_prompt"],
                text_id,
                union_errors,
                diag_length,
                diag_token_mismatch,
            )

    summary = {
        "prompts_processed": max(0, end - config["start_prompt"]),
        "rows_union_masks_value_error": union_errors,
        "rows_length_mismatch_in_diagnose": diag_length,
        "rows_token_id_mismatch_in_diagnose": diag_token_mismatch,
        "failure_examples": examples[: config["max_failure_examples_saved"]],
    }

    os.makedirs(os.path.dirname(config["report_json"]) or ".", exist_ok=True)
    with open(config["report_json"], "w") as rf:
        json.dump(summary, rf, indent=2)
    logger.info("Wrote report to %s", config["report_json"])
    logger.info(
        "Summary: processed=%s union ValueError=%s length_mismatch=%s token_id_mismatch=%s",
        summary["prompts_processed"],
        summary["rows_union_masks_value_error"],
        summary["rows_length_mismatch_in_diagnose"],
        summary["rows_token_id_mismatch_in_diagnose"],
    )

    problematic = union_errors > 0 or diag_token_mismatch > 0
    return 1 if problematic else 0


def _build_config_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    c = vars(args).copy()
    c["build_loss_dict"] = {
        "length_normalize": True,
        "alpha": 1.0,
        "AR_temperature": 1.0,
        "AR_top_k": 0,
        "AR_top_p": 0.96,
        "max_output_length": 20,
    }
    report_path = c.pop("report_json")
    max_prompts = c.pop("max_prompts")
    start_prompt = c.pop("start_prompt")
    log_every = c.pop("log_every")
    max_failure_examples_saved = c.pop("max_failure_examples_saved")
    c["report_json"] = report_path
    c["max_prompts"] = max_prompts
    c["start_prompt"] = start_prompt
    c["log_every"] = log_every
    c["max_failure_examples_saved"] = max_failure_examples_saved
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Locate + union_masks only — check MLM-encoded length / token-id consistency."
    )
    parser.add_argument("--task", type=str, required=True, help='Use underscore for multi, e.g. "toxicity_sentiment".')
    parser.add_argument("--source_data", type=str, required=True)
    parser.add_argument("--target_label_ids", nargs="+", type=int, required=True)
    parser.add_argument("--model_paths", nargs="+", type=str, required=True)
    parser.add_argument("--tokenizer_paths", nargs="+", type=str, required=True)
    parser.add_argument("--model_types", nargs="+", type=str, required=True)
    parser.add_argument("--losses", nargs="+", type=str, required=True, help="len > 2 (LM + 2+ aux).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "/tmp/hf_cache"))
    parser.add_argument("--jsonl_primary_key", type=str, default="prompt")
    parser.add_argument("--jsonl_secondary_key", type=str, default="text")
    parser.add_argument("--locate_method", type=str, choices=["attention", "grad_norm"], default="attention")
    parser.add_argument("--num_edit_token_per_step", type=int, default=7)
    parser.add_argument("--locate_unit", type=str, default="token")
    parser.add_argument("--report_json", type=str, default="locate_union_masks_report.json")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--start_prompt", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_failure_examples_saved", type=int, default=50)

    cli_args = parser.parse_args()
    if len(cli_args.losses) <= 2:
        parser.error("Need more than two --losses entries for union_masks (e.g. gpt2 + clf1 + clf2).")

    exit_code = run_check(_build_config_from_cli(cli_args))
    raise SystemExit(exit_code)
