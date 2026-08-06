# LaSEr Edit — Running Editing Methods

This guide explains how to run the main locate/edit pipelines for each **method × task** pair.

**Preferred entrypoints** (one script per setting) live under:

```text
laser_edit/_bash_scripts/entrypoints/
```

All commands assume you are at the **repository root** (the parent of `laser_edit/`), with `PYTHONPATH=.`.

```bash
cd /path/to/mucoco
export PYTHONPATH=.
```

---

## Environment setup

Create **two** conda environments. Do not try to install vLLM into the CUDA 12.4 stack — its torch/CUDA pins conflict with that freeze.

| Environment | Python / CUDA | Script | Requirements | Includes vLLM? |
| --- | --- | --- | --- | --- |
| `loc-edit` | 3.11 / CUDA 12.4 | `create_env_py3.11_cuda12.4.sh` | `requirements_py3.11_cuda12.4.txt` | **No** |
| `loc-edit-pro6000-vllm` | 3.12 / CUDA 13.0 | `create_env_py3.12_cuda13.0_vllm.sh` | `requirements_py3.12_cuda13.0_vllm.txt` | **Yes** |

From `laser_edit/`:

```bash
# 1) General / EBM stack (no vLLM)
bash create_env_py3.11_cuda12.4.sh

# 2) LLM paths that need vLLM (CUDA 13.0 only)
bash create_env_py3.12_cuda13.0_vllm.sh
```

**vLLM:** only install via the CUDA 13.0 + Python 3.12 script above. Adding vLLM to `create_env_py3.11_cuda12.4.sh` / `requirements_py3.11_cuda12.4.txt` does not resolve cleanly (exact torch and dependency pins clash). Use `loc-edit-pro6000-vllm` (or an existing dedicated `vllm` env) for `--use_vllm` jobs.

---

## Method overview

| Method | Locate | Edit |
| --- | --- | --- |
| **Plain LLM Edit** | none (edit the full text) | LLM edit |
| **Self-locate & LLM Edit** | same LLM locates spans | LLM edit |
| **Self-parallel-locate & LLM Edit** | two separate LLM locates (toxicity + inconsistency), then union | LLM edit |
| **LaSEr & LLM Edit** | LaSEr (energy-based) locate | LLM edit |
| **LaSEr & EBM Edit** | LaSEr locate | energy-based (EBM) decoding edit |
| **LaSEr & EBM Edit + LLM smoothing** | same as LaSEr & EBM edit | same as LaSEr & EBM edit → applies additional LLM smoothing |

### Tasks

Nicknames for each task is in the parentheses.
- **toxicity avoidance (toxicity)**: enforcing nontoxicity
- **contradiction avoidance (nli)**: enforcing single-pair logical consistency (NLI-style)
- **set-consistency enforcement (set-LConVQA)**: enforcing consistency across a set of question-answer pairs
- **set-consistency enforcement (set-SNLI)**: enforcing consistency across a set of sentences
- **joint toxicity and contradiction avoidance (multi)**: jointly enforcing both nontoxicity and single-pair logical consistency

---

## Quick start (dedicated entrypoints)

| Method | Task | Submit |
| --- | --- | --- |
| Plain LLM Edit | toxicity | `sbatch laser_edit/_bash_scripts/entrypoints/plain_llm/toxicity.sh` |
| Plain LLM Edit | nli | `sbatch laser_edit/_bash_scripts/entrypoints/plain_llm/nli.sh` |
| Plain LLM Edit | set-LConVQA | `sbatch …/plain_llm/set_lconvqa.sh` |
| Plain LLM Edit | set-NLI (set-SNLI) | `sbatch …/plain_llm/set_nli.sh` |
| Plain LLM Edit | multi | `sbatch …/plain_llm/multi.sh` |
| Self-locate & LLM Edit | toxicity | `…/self_locate_llm/toxicity_{locate,postprocess,edit}.sh` (in order) |
| Self-locate & LLM Edit | nli | `…/self_locate_llm/nli_{locate,postprocess,edit}.sh` |
| Self-locate & LLM Edit | set-LConVQA | `…/self_locate_llm/set_lconvqa_{locate,edit}.sh` |
| Self-locate & LLM Edit | set-NLI (set-SNLI) | `…/self_locate_llm/set_nli_{locate,edit}.sh` |
| Self-locate & LLM Edit | multi | `…/self_locate_llm/multi_{locate,postprocess,edit}.sh` |
| Self-parallel-locate & LLM Edit | multi | `…/self_parallel_locate_llm/multi_{locate,postprocess,edit}.sh` |
| LaSEr & LLM Edit | toxicity | `sbatch …/laser_llm/toxicity.sh` |
| LaSEr & LLM Edit | nli | `sbatch …/laser_llm/nli.sh` |
| LaSEr & LLM Edit | set-LConVQA | `…/laser_llm/set_lconvqa_{locate,edit}.sh` |
| LaSEr & LLM Edit | set-NLI (set-SNLI) | `…/laser_llm/set_nli_{locate,edit}.sh` |
| LaSEr & LLM Edit | multi | `sbatch …/laser_llm/multi.sh` |
| LaSEr & EBM Edit | toxicity | `sbatch …/laser_ebm/toxicity.sh` |
| LaSEr & EBM Edit | nli | `sbatch …/laser_ebm/nli.sh` |
| LaSEr & EBM Edit | set-LConVQA | `sbatch …/laser_ebm/set_lconvqa.sh` |
| LaSEr & EBM Edit | set-NLI (set-SNLI) | `sbatch …/laser_ebm/set_nli.sh` |
| LaSEr & EBM Edit | multi | `sbatch …/laser_ebm/multi.sh` |
| + LLM smoothing | toxicity / nli / set-LConVQA / set-NLI / multi | `sbatch …/laser_ebm_llm_smoothing/<task>.sh` (update `INPUT_PATH` first) |

For multi-step pipelines, run **locate → postprocess → edit** in order. After locate finishes, update result filenames in the postprocess/edit scripts (they include timestamps / job ids).
