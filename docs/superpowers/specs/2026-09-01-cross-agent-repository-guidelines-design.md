# Cross-Agent Repository Guidelines Design

**Date:** 2026-09-01

## Purpose

Create a repository-root `AGENTS.md` that gives Codex, Cursor, Claude, and other coding agents the same practical operating rules for LaSEr-Edit. The guidance must balance four equally important goals: reproducibility, protection of existing research results, fast iteration, and code quality.

The file will describe repository-wide expectations without depending on commands or features unique to one agent product.

## Scope and Precedence

- The root `AGENTS.md` applies to the entire repository.
- A nested `AGENTS.md` may add or override rules for its subtree.
- Instructions are followed in this order: the user's current request and safety constraints; protection of user work and research artifacts; reproducibility and correctness; verifiable code quality; speed and convenience.
- If instructions conflict or a decision could materially change experimental meaning, agents should state the conflict and ask before proceeding.

## Repository Context

The guidelines will briefly identify this as the LaSEr-Edit research codebase for energy-based localization and editing, including toxicity, NLI, set-consistency, and multi-task workflows. They will point agents toward the existing project documentation, configurations, and experiment entrypoints rather than duplicating detailed command documentation.

Expected conventions include:

- Run project commands from the repository root with `PYTHONPATH=.` unless the relevant entrypoint says otherwise.
- Prefer maintained scripts under `laser_edit/_bash_scripts/entrypoints/` and existing configuration patterns.
- Use the documented Conda environments, including `loc-edit` for general/EBM work and `loc-edit-pro6000-vllm` for vLLM workflows, when applicable.

## Standard Workflow

Before changing files, an agent should:

1. Read the applicable `AGENTS.md` files and relevant local documentation.
2. Inspect `git status` and preserve all existing modified and untracked files.
3. Inspect the relevant entrypoints, configurations, and implementation before proposing or making changes.
4. Keep work tightly scoped; avoid opportunistic refactors and unrelated formatting changes.

Requests to analyze, diagnose, review, or report do not by themselves authorize code or data changes. When implementation is requested, agents may make the smallest coherent change and perform proportionate verification. Clarification is reserved for ambiguities whose resolution would materially affect behavior, data, compute cost, or experimental interpretation.

Secrets, access tokens, private endpoints, and credentials must never be copied into code, logs, documentation, or responses.

## Experiment Reproducibility and Artifact Safety

For experiments and hyperparameter tuning, agents should record enough information to reproduce and compare runs:

- task and dataset or subset;
- model and checkpoint identifiers;
- random seed and input ordering;
- decoding and generation settings;
- tuned hyperparameters and fixed controls;
- command or submission script, environment, output path, and relevant code revision.

Comparisons should use fixed inputs, ordering, seeds, and evaluation procedures whenever feasible. Hyperparameters should be selected on development or validation data; test data should be reserved for final evaluation unless the user explicitly defines a different protocol. Interpretation must account for leakage, seed differences, sample-size differences, and incomparable configurations.

Datasets, checkpoints, generated outputs, evaluation artifacts, W&B records, and logs are treated as valuable research assets. New runs should use unique, descriptive output paths. Agents must not overwrite, delete, rename, or move existing artifacts unless the user explicitly requests it and the exact targets have been verified.

## SLURM-Only Verification Policy

All runtime tests and verification must be executed as SLURM jobs. This includes unit tests, CPU-only tests, smoke tests, integration tests, evaluations, and experiment runs. Agents should use `sbatch` or the cluster-approved form of `srun`, following repository and cluster conventions.

Local or login-node execution should be avoided. It is allowed only when the user explicitly requests it or when a minimal static diagnostic is necessary to prepare or troubleshoot a SLURM submission. Such a local diagnostic is never sufficient evidence that a change is complete or correct.

A runtime verification counts as successful only when:

1. the SLURM job reaches a successful terminal state;
2. its stdout and stderr logs have been inspected;
3. the expected assertions, metrics, outputs, or behaviors are confirmed.

A queued, running, failed, timed-out, or cancelled job means verification remains incomplete. Completion reports must include the job ID, submission script or command, final status, log path, and verified result. Static checks that do not execute project code may be performed locally when appropriate, but they do not replace required SLURM runtime verification.

## Code Quality and Error Handling

- Behavior changes and bug fixes should have a reproducible test or check that would detect regression.
- When practical, demonstrate the failure first, then make the minimum implementation change needed to pass the check.
- Validate configurations, paths, shapes, and CLI arguments early and fail with actionable messages.
- Do not swallow exceptions, silently skip failed samples, fabricate results, or report success after partial failure.
- Logs should identify the task, dataset, model, seed, key settings, and output location without exposing secrets.
- Follow existing interfaces and patterns unless changing them is necessary and explicitly justified.

## Git and File Safety

- Inspect `git status` before and after work.
- Treat existing modifications and untracked files as user-owned.
- Do not use destructive restoration, broad deletion, forced checkout, or history-rewriting commands.
- Stage only files belonging to the current task, and do not create commits unless requested or required by an agreed workflow.
- Never bundle unrelated user changes into a commit.
- Before destructive artifact operations, resolve and verify exact paths and explain recoverability.

## Completion Reporting

Agents should lead with the outcome and provide evidence. A completion report should identify:

- files and behavior changed;
- SLURM job ID, final status, log path, and verification result;
- tests or evaluations not run and why;
- remaining risks or assumptions;
- research data or artifacts generated, moved, overwritten, or deleted.

Analysis-only work should cite the inspected evidence and clearly distinguish recommendations from implemented changes. No task should be described as complete or passing without corresponding verification evidence.

## Proposed `AGENTS.md` Structure

The final file will use concise, imperative sections in this order:

1. Scope and priorities
2. Repository orientation
3. Working rules
4. Experiments and reproducibility
5. SLURM testing and verification
6. Code quality and error handling
7. Git, files, and artifact safety
8. Completion reporting

This structure keeps universal rules easy to scan while retaining the LaSEr-Edit-specific operational details needed for reliable research work.

## Non-Goals

- Replacing the project's README, environment files, or experiment documentation.
- Encoding one-off hyperparameter values or task-specific procedures that will quickly become stale.
- Adding tool-specific behavior for a single coding assistant.
- Requiring broad refactors, new infrastructure, or changes to existing experiments.

## Acceptance Criteria

The resulting root `AGENTS.md` is acceptable when it:

- applies consistently across coding-agent products;
- reflects the repository's actual execution and environment conventions;
- gives equal weight to reproducibility, artifact protection, iteration speed, and code quality;
- explicitly requires all runtime testing and verification to run through SLURM;
- discourages local and login-node testing and defines the narrow exception;
- defines what evidence is required before reporting completion;
- protects existing dirty-worktree changes and research outputs;
- remains concise enough to be read at the start of routine tasks.
