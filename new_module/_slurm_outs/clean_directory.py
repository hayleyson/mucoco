#!/usr/bin/env python3
"""Delete *.out files matching traceback, Slurm/srun failure patterns (optional max lines)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

# Standard first line of a Python traceback (strict match).
_TRACEBACK_HEADER = "Traceback (most recent call last):"

# e.g. slurmstepd-n02: error: *** JOB 142524 ON n02 CANCELLED AT 2026-05-06T13:38:33 ***
_SLURM_JOB_CANCELLED = re.compile(
    r"slurmstepd-([^:]+): error: \*\*\* JOB (\d+) ON \1 CANCELLED AT"
)

# e.g. slurmstepd-n02: error: *** STEP 142524.0 ON n02 CANCELLED AT 2026-05-06T13:38:33 ***
# (.batch covers Slurm's batch script step id)
_SLURM_STEP_CANCELLED = re.compile(
    r"slurmstepd-([^:]+): error: \*\*\* STEP (\d+\.(?:\d+|batch)) ON \1 CANCELLED AT"
)

# e.g. srun: error: n03: task 0: Exited with exit code 2
_SRUN_TASK_NONZERO_EXIT = re.compile(
    r"srun:\s*error:\s*[^:]+:\s*task\s+\d+:\s*Exited with exit code\s+[1-9]\d*",
    re.IGNORECASE,
)

# e.g. srun: error: Unable to create step for job 120355: More processors requested than permitted
_SRUN_UNABLE_CREATE_STEP = re.compile(
    r"srun:\s*error:\s*Unable to create step for job\s+\d+:",
    re.IGNORECASE,
)


def _file_has_traceback(text: str) -> bool:
    return _TRACEBACK_HEADER in text


def _file_has_slurm_job_cancelled(text: str) -> bool:
    return _SLURM_JOB_CANCELLED.search(text) is not None


def _file_has_slurm_step_cancelled(text: str) -> bool:
    return _SLURM_STEP_CANCELLED.search(text) is not None


def _file_has_srun_task_nonzero_exit(text: str) -> bool:
    return _SRUN_TASK_NONZERO_EXIT.search(text) is not None


def _file_has_srun_unable_create_step(text: str) -> bool:
    return _SRUN_UNABLE_CREATE_STEP.search(text) is not None


def _file_matches_delete_criteria(text: str) -> bool:
    return (
        _file_has_traceback(text)
        or _file_has_slurm_job_cancelled(text)
        or _file_has_slurm_step_cancelled(text)
        or _file_has_srun_task_nonzero_exit(text)
        or _file_has_srun_unable_create_step(text)
    )


def _line_count(text: str) -> int:
    return len(text.splitlines())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan *.out files in the script directory and remove those whose contents "
            "include a Python traceback, a Slurm slurmstepd 'JOB' / 'STEP' "
            "... CANCELLED AT line, srun 'Exited with exit code' (non-zero), or "
            "srun 'Unable to create step for job'. "
            "With --max-lines, only files shorter than N lines are removed."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list paths that would be deleted; do not remove files.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only delete if the file has fewer than N lines (in addition to matching a "
            "traceback, Slurm cancellation, or srun failure lines). If omitted, line count is not checked."
        ),
    )
    args = parser.parse_args()

    targets: list[Path] = []
    for path in sorted(_THIS_DIR.glob("*.out")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"skip read error {path.name}: {exc}", file=sys.stderr)
            continue
        if not _file_matches_delete_criteria(text):
            continue
        if args.max_lines is not None and _line_count(text) >= args.max_lines:
            continue
        targets.append(path)

    for path in targets:
        if args.dry_run:
            print(f"would delete: {path.name}")
        else:
            try:
                path.unlink()
                print(f"deleted: {path.name}")
            except OSError as exc:
                print(f"failed to delete {path.name}: {exc}", file=sys.stderr)

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action} {len(targets)} .out file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
