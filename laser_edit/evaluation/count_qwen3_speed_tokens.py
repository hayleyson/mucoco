import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def output_texts(rows, set_consistency_llm_edit_output: bool):
    if set_consistency_llm_edit_output:
        for row in rows:
            for pred in row.get("pred", []):
                yield " ".join(iter_strings(pred))
    else:
        for row in rows:
            for gen in row.get("generations", []):
                yield gen.get("text", "")


def time_file_candidates(path: Path):
    yield Path(str(path) + ".time")
    yield path.with_name(path.name.replace("_edit_result.jsonl", "_edit_metrics.jsonl"))


def read_seconds(path: Path) -> float:
    for candidate in time_file_candidates(path):
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if candidate.name.endswith("_edit_metrics.jsonl"):
            metrics = json.loads(text)
            return float(
                metrics.get(
                    "total_time_seconds",
                    metrics.get("time_seconds-1-incon", 0.0),
                )
            )

        values = {}
        for line in text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                try:
                    values[key.strip()] = float(value.strip())
                except ValueError:
                    pass
        if "llm_edit_seconds" in values:
            return values["llm_edit_seconds"]
        if "total_elapsed_seconds" in values:
            return values["total_elapsed_seconds"]
        stripped = text.strip()
        if stripped:
            try:
                return float(stripped.splitlines()[0])
            except ValueError:
                pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations_file_path", required=True)
    parser.add_argument("--results_file_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--set_consistency_llm_edit_output", action="store_true")
    args = parser.parse_args()

    generations_path = Path(args.generations_file_path)
    results_path = Path(args.results_file_path)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    rows = read_jsonl(generations_path)
    texts = list(output_texts(rows, args.set_consistency_llm_edit_output))
    num_outputs = len(texts)
    num_tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
    )
    seconds = read_seconds(generations_path)
    tokens_per_second = num_tokens / seconds if seconds > 0 else 0.0
    outputs_per_second = num_outputs / seconds if seconds > 0 else 0.0

    with results_path.open("w", encoding="utf-8") as f:
        f.write(f"speed_num_outputs: {num_outputs}\n")
        f.write(f"speed_qwen3_tokens: {num_tokens}\n")
        f.write(f"speed_seconds: {seconds}\n")
        f.write(f"speed_qwen3_tokens_per_second: {tokens_per_second}\n")
        f.write(f"speed_outputs_per_second: {outputs_per_second}\n")
        f.write(f"speed_task: {args.task}\n")


if __name__ == "__main__":
    main()
