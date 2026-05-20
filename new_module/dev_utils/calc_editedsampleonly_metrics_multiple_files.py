"""
이런 코드가 많이 필요한게 좋은건지 모르겠지만,
특정 index의 sample만 가지고 결과를 다시 계산하는 코드

여러 output 파일을 순회하고, 지정한 출력 디렉터리에 하나의 집계 CSV로 저장한다.
"""

import argparse
import math
import os
from glob import glob

import pandas as pd

os.chdir("/home/hyeryung/data/mucoco")
from evaluation.prompted_sampling.evaluate import distinctness
from new_module.dev_utils.utils import read_metric_file


def unravel(outputs_df):
    outputs_df = outputs_df.explode("generations", ignore_index=True)

    outputs_df["prompt"] = outputs_df["prompt"].apply(lambda x: x["text"])

    outputs_df["text"] = outputs_df["generations"].apply(lambda x: x["text"])

    gen_dict = outputs_df["generations"].values[0]

    for col in gen_dict.keys():
        outputs_df[col] = outputs_df["generations"].apply(lambda x: x.get(col, None))

    return outputs_df


def reformat(unraveled_df):
    if "tokens" in unraveled_df:
        unraveled_df["generations"] = unraveled_df.apply(
            lambda x: [{"text": x["text"], "tokens": x["tokens"]}], axis=1
        )
    else:
        unraveled_df["generations"] = unraveled_df.apply(
            lambda x: [{"text": x["text"]}], axis=1
        )
    return_df = unraveled_df.copy()
    return_df["prompt"] = return_df["prompt"].apply(lambda x: {"text": x})

    return return_df


def ravel(unraveled_df):
    if "tokens" in unraveled_df:
        unraveled_df["generations"] = unraveled_df.apply(
            lambda x: [{"text": x["text"], "tokens": x["tokens"]}], axis=1
        )
    else:
        unraveled_df["generations"] = unraveled_df.apply(
            lambda x: [{"text": x["text"]}], axis=1
        )
    return_df = unraveled_df.groupby("prompt")["generations"].sum([]).reset_index()
    return_df["prompt"] = return_df["prompt"].apply(lambda x: {"text": x})

    return return_df


def unravel_toxicity_data(df):
    df["toxicity"] = df["allresponses"].apply(
        lambda x: [
            x[0]["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            for x in list(x.values())
        ]
    )
    df = df.explode("toxicity", ignore_index=True)
    return df


def build_edited_ixs(index_files, nicknames):
    edited_ixs = {}
    if len(index_files) == 1:
        with open(index_files[0], "r") as f:
            raw = f.read()
        indexes = [int(x) for x in raw.strip().split() if x]
        for n in nicknames:
            edited_ixs[n] = indexes
    else:
        if len(index_files) != len(nicknames):
            raise ValueError(
                "index_files must have length 1 (shared) or the same length as nicknames"
            )
        for idx_path, n in zip(index_files, nicknames):
            with open(idx_path, "r") as f:
                raw = f.read()
            edited_ixs[n] = [int(x) for x in raw.strip().split() if x]
    return edited_ixs


def resolve_result_file_prefix(output_path):
    if len(glob(f"{output_path}-results.txt.*")) > 0:
        return f"{output_path}-results.txt"
    alt = output_path.replace("/outputs_", "/results_").replace(".txt", "-test.txt")
    if len(glob(alt)) > 0:
        return alt
    return f"{output_path}-results.txt"


def resolve_fluency_result_path(output_path: str, result_file_prefix: str) -> str:
    """Return path to .fluency sidecar; may live under parallel fluency/ rather than final/."""
    if "saeheeeom" in output_path:
        base = (
            result_file_prefix.replace("/final/", "/final_fluency/")
            .replace("/edited/", "/edited_fluency/")
        )
    else:
        base = result_file_prefix
    path = base + ".fluency"
    if os.path.isfile(path):
        return path
    alt = base.replace("/final/", "/fluency/") + ".fluency"
    if os.path.isfile(alt):
        return alt
    return path


def task_metrics_list(task, sbert):
    if task == "toxicity":
        metrics = [
            "fluency",
            "ppl-big-qwen",
            "repetitions",
            "toxicity",
            "toxicity_int",
            "dist-3",
        ]
    elif task == "sentiment":
        metrics = ["fluency", "ppl-big-qwen", "repetitions", "sentiment_ext", "dist-3"]
    elif task == "nli":
        metrics = ["fluency", "ppl-big-qwen", "repetitions", "nli", "dist-3"]
    else:
        raise ValueError(f"Unknown task: {task}")
    if sbert:
        metrics = list(metrics) + ["sbert"]
    return metrics


def compute_metrics_row(output_path, suffix, edited_ixs, metrics, task):
    """Compute one row of metrics for a single output jsonl and nickname."""
    nicknames = [suffix]
    output_files = [output_path]
    result_file_prefix = resolve_result_file_prefix(output_path)

    print(f"Processing nickname={suffix} output={output_path}")

    metric = "ppl-big-qwen"
    ppl_qwen_metrics = []
    total_ppl_qwen_metrics = []
    result_files = [result_file_prefix + ".ppl-big-qwen"]
    result_files_alt = []

    for i, s in enumerate(nicknames):
        result_file = [result_files[i]]
        if metric in ["repetitions", "toxicity"]:
            try:
                result = pd.read_json(result_file[0], lines=True, dtype=float)
            except Exception:
                result = pd.read_json(result_files_alt[i], lines=True, dtype=float)
        else:
            try:
                result = pd.read_csv(result_file[0], header=None, dtype=float)
            except Exception:
                result = pd.read_csv(result_files_alt[i], header=None, dtype=float)
        print(result.dtypes)
        result = result.loc[edited_ixs[s]]
        ppl_qwen_metrics.append(result[0].mean())
        metric_value = math.exp(float(result[1].sum()) / float(result[2].sum()))
        total_ppl_qwen_metrics.append(metric_value)

    metric = "fluency"
    fluency_metrics = []
    fluency_file = resolve_fluency_result_path(output_files[0], result_file_prefix)
    result_files = [fluency_file]

    for i, s in enumerate(nicknames):
        result_file = [result_files[i]]
        if metric in ["repetitions", "toxicity"]:
            result = pd.read_json(result_file[0], lines=True)
        else:
            result = pd.read_csv(result_file[0], header=None)

        result = result.loc[edited_ixs[s]]
        if "LABEL_1" in result[0].unique():
            metric_value = result.loc[result[0] == "LABEL_1"].shape[0] / result.shape[0]
        else:
            metric_value = result.loc[result[0] == 1].shape[0] / result.shape[0]
        fluency_metrics.append(metric_value)

    metric = "repetitions"
    repetitions_metrics = []
    result_files = [result_file_prefix + ".repetitions"]

    for i, s in enumerate(nicknames):
        result_file = [result_files[i]]
        if metric in ["repetitions", "toxicity"]:
            result = pd.read_json(result_file[0], lines=True)
        else:
            result = pd.read_csv(result_file[0], header=None)

        result = result.loc[edited_ixs[s]]
        if result.empty:
            metric_value = 0
        else:
            metric_value = (
                result.loc[~result["repeated_phrase"].isna(), :].shape[0]
                / result.shape[0]
            )
        repetitions_metrics.append(metric_value)

    dist3_metrics = []
    for i, s in enumerate(nicknames):
        output_file = [output_files[i]]
        outputs = pd.read_json(output_file[0], lines=True)
        outputs = unravel(outputs)
        outputs = outputs.loc[edited_ixs[s]]
        outputs = ravel(outputs)
        _, _, dist3 = distinctness(outputs)
        dist3_metrics.append(dist3)

    avg_max_toxicity_metrics = []
    avg_toxicity_metrics = []
    toxic_proba_metrics = []
    toxic_proba_75_metrics = []
    positive_proba_ext_metrics = []
    contra_proba_metrics = []

    if "toxicity" in metrics:
        metric = "toxicity"
        result_files = [result_file_prefix + ".toxicity"]

        for i, s in enumerate(nicknames):
            result_file = [result_files[i]]
            if metric in ["repetitions", "toxicity"]:
                result = pd.read_json(result_file[0], lines=True)
            else:
                result = pd.read_csv(result_file[0], header=None)

            if result.empty:
                result_file = glob(
                    f"outputs/toxicity/**/**/*{s}*/results_filled.txt.{metric}"
                )
                if len(result_file) == 0:
                    result_file = glob(
                        f"outputs/toxicity/**/*{s}*/results_filled.txt.{metric}"
                    )
                if metric in ["repetitions", "toxicity"]:
                    result = pd.read_json(result_file[0], lines=True)
                else:
                    result = pd.read_csv(result_file[0], header=None)
            result = unravel_toxicity_data(result)
            result = result.loc[edited_ixs[s]]
            avg_toxicity = result["toxicity"].mean()
            toxic_proba = result["toxicity"].apply(lambda x: 1 if x > 0.5 else 0).mean()
            toxic_proba_75 = (
                result["toxicity"].apply(lambda x: 1 if x > 0.75 else 0).mean()
            )

            outputs = pd.read_json(output_files[i], lines=True)
            outputs = unravel(outputs)
            outputs = outputs.loc[edited_ixs[s]]
            outputs_result = pd.concat([outputs, result], axis=1)
            avg_max_toxicity = outputs_result.groupby("prompt")["toxicity"].max().mean()

            avg_max_toxicity_metrics.append(avg_max_toxicity)
            avg_toxicity_metrics.append(avg_toxicity)
            toxic_proba_metrics.append(toxic_proba)
            toxic_proba_75_metrics.append(toxic_proba_75)

    if "sentiment_ext" in metrics:
        metric = "sentiment_ext"
        result_files = [result_file_prefix + f".{metric}"]
        for i, s in enumerate(nicknames):
            result_file = result_files[i]
            result = pd.read_json(result_file, lines=True)
            result = result.loc[edited_ixs[s]]
            metric_value = result["label"].apply(
                lambda x: 1 if x == "POSITIVE" else 0
            ).mean()
            positive_proba_ext_metrics.append(metric_value)

    if "nli" in metrics:
        metric = "nli"
        result_files = [result_file_prefix + f".{metric}"]
        for i, s in enumerate(nicknames):
            result_file = result_files[i]
            result = read_metric_file(result_file, "nli")
            result = result.loc[edited_ixs[s]]
            metric_value = result["nli_class"].apply(
                lambda x: 1 if x == "contradiction" else 0
            ).mean()
            contra_proba_metrics.append(metric_value)

    sbert_metrics = []
    sbert_geq_5_counts = []
    sbert_geq_5_ratios = []
    if "sbert" in metrics:
        for s in nicknames:
            result_files = [result_file_prefix + ".sbertscore"]
            if not os.path.exists(result_files[0]):
                continue
            with open(result_files[0], "r") as f:
                raw_data = f.readlines()
                tmp_data = []
                for x in raw_data[1:]:
                    try:
                        tmp_data.append(float(x.strip()))
                    except Exception:
                        tmp_data.append(float("nan"))

            result = pd.DataFrame({"sbert": tmp_data})
            result = result.loc[edited_ixs[s]]
            sbert_score = result.sbert.mean()
            sbert_count = result.loc[result.sbert >= 0.5].shape[0]
            sbert_ratio = sbert_count / result.shape[0]
            sbert_metrics.append(sbert_score)
            sbert_geq_5_counts.append(sbert_count)
            sbert_geq_5_ratios.append(sbert_ratio)

    row = {
        "output_file": output_path,
        "nickname": f"llm_gens_{suffix}",
        "delta_ppl": "",
        "fluency_metrics": fluency_metrics[0],
        "dist-3": dist3_metrics[0],
        "rep_rate": repetitions_metrics[0],
        "num_edits": len(edited_ixs[suffix]),
    }

    if "sbert" in metrics and len(sbert_metrics) > 0:
        row["sbert"] = sbert_metrics[0]
        row["sbert_count"] = sbert_geq_5_counts[0]
        row["sbert_ratio"] = sbert_geq_5_ratios[0]
    else:
        row["sbert"] = ""
        row["sbert_count"] = ""
        row["sbert_ratio"] = ""

    if task == "toxicity":
        row["ppl_qwen"] = ppl_qwen_metrics[0]
        row["total_ppl_qwen"] = total_ppl_qwen_metrics[0]
        row["avg_max_toxicity"] = avg_max_toxicity_metrics[0]
        row["avg_toxicity"] = avg_toxicity_metrics[0]
        row["toxic_proba"] = toxic_proba_metrics[0]
        row["toxic_75_proba"] = toxic_proba_75_metrics[0]
    elif task == "sentiment":
        row["sentiment_ext"] = positive_proba_ext_metrics[0]
        row["ppl"] = ppl_qwen_metrics[0]
        row["total_ppl"] = total_ppl_qwen_metrics[0]
    elif task == "nli":
        row["contra_prob"] = contra_proba_metrics[0]
        row["ppl"] = ppl_qwen_metrics[0]
        row["total_ppl"] = total_ppl_qwen_metrics[0]

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read raw metric files and calculate summary statistics only using a selected set of samples. "
        "Loops over output files and writes one aggregated CSV."
    )
    parser.add_argument(
        "--output_files",
        nargs="+",
        type=str,
        help="Output file paths, e.g. <path>/5_tox_loc_edit_38576.jsonl",
    )
    parser.add_argument(
        "--index_files",
        nargs="+",
        type=str,
        help="Either one shared index file (applied to every run), or one index file per run. "
        "If more than one file is passed, counts must match --nicknames and --output_files.",
    )
    parser.add_argument(
        "--nicknames",
        nargs="+",
        type=str,
        help="Run ids. With a single --index_file, pass one nickname (repeated automatically "
        "for each --output_file) or one per output. With multiple --index_files, "
        "nickname i pairs with index_files[i]. Each nickname keys edited_ixs; with one "
        "shared index file, every name shares the same index list. Use distinct names "
        "per output if toxicity path globs must differ. For task toxicity, substring "
        "under outputs/toxicity/... when the primary .toxicity file is empty.",
    )
    parser.add_argument("--task", type=str, help="toxicity | sentiment | nli")
    parser.add_argument(
        "--sbert",
        action="store_true",
        help="Include bertscore in the summary statistics",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where the aggregated CSV is written",
    )
    parser.add_argument(
        "--aggregate_csv",
        type=str,
        default="aggregated_editedsampleonly_metrics.csv",
        help="Filename for the aggregated CSV (inside --output_dir)",
    )
    args = parser.parse_args()
    if not args.task:
        parser.error("--task is required (toxicity | sentiment | nli)")

    n_out = len(args.output_files)
    if len(args.index_files) == 1:
        if len(args.nicknames) == 1 and n_out > 1:
            args.nicknames = [args.nicknames[0]] * n_out
        if len(args.nicknames) != n_out:
            raise ValueError(
                "With a single --index_file, pass one --nickname (reused for every "
                "--output_file) or one nickname per output file (same length as --output_files)."
            )
    else:
        if len(args.index_files) != len(args.nicknames):
            raise ValueError(
                "With multiple --index_files, pass one --nickname per index file "
                "(same length as --index_files and --output_files)."
            )
        if len(args.index_files) != n_out:
            raise ValueError(
                "With multiple --index_files, --output_files must have the same length "
                "so each output pairs with its index file."
            )

    edited_ixs = build_edited_ixs(args.index_files, args.nicknames)
    metrics = task_metrics_list(args.task, args.sbert)

    rows = []
    for out_path, suffix in zip(args.output_files, args.nicknames):
        rows.append(
            compute_metrics_row(out_path, suffix, edited_ixs, metrics, args.task)
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, args.aggregate_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote aggregated metrics to {out_csv}")
