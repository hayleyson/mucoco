#!/usr/bin/env python3
"""Map each (prompt, generation) in th0.39 JSONL to 0-based exploded index in th0.95 JSONL."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

BASE = Path(__file__).resolve().parent
TH048 = BASE / "gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_edit_candidates_0_39.jsonl"
TH3105 = BASE / "gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_below_nontoxic_threshold_0_95_332.jsonl"
OUT = BASE / "gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_edit_candidates_0_39.indexes_in_0_95.txt"


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_exploded_queues(path: Path) -> dict[tuple[str, str], deque[int]]:
    """Explode generations in file order; 0-based global index. Queue per (prompt, gen) for duplicates."""
    key_to_indices: dict[tuple[str, str], deque[int]] = {}
    idx = 0
    for obj in iter_jsonl(path):
        p = obj["prompt"]["text"]
        for g in obj["generations"]:
            key = (p, g["text"])
            if key not in key_to_indices:
                key_to_indices[key] = deque()
            key_to_indices[key].append(idx)
            idx += 1
    return key_to_indices


def main():
    queues = build_exploded_queues(TH3105)
    indexes: list[int] = []
    missing: list[tuple[str, str, int]] = []  # key + line hint

    line_no = 0
    for obj in iter_jsonl(TH048):
        line_no += 1
        p = obj["prompt"]["text"]
        for g in obj["generations"]:
            key = (p, g["text"])
            q = queues.get(key)
            if not q:
                missing.append((p, g["text"], line_no))
                continue
            indexes.append(q.popleft())

    if missing:
        preview = missing[:3]
        raise SystemExit(
            f"No match in th0.95 file for {len(missing)} (prompt, generation) pair(s). "
            f"First few: line nos {[m[2] for m in preview]} ..."
        )

    OUT.write_text(" ".join(str(i) for i in indexes) + "\n")
    print(f"Wrote {len(indexes)} indexes to {OUT}")


if __name__ == "__main__":
    main()
