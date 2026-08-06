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

Create **two** conda environments by running the commands below from `laser_edit/`:

```bash
# 1) General / EBM stack (no vLLM)
bash create_env_py3.11_cuda12.4.sh

# 2) LLM paths that need vLLM (CUDA 13.0 only)
bash create_env_py3.12_cuda13.0_vllm.sh
```

Currently, only `create_env_py3.12_cuda13.0_vllm.sh` installs vLLM. Use `loc-edit-pro6000-vllm` if you want to run LLM edit experiments with the `--use_vllm` flag.

---

## Downloading energy models (EBMs)

LaSEr locate / EBM edit need **local** energy-model checkpoints.

### Toxicity and NLI

Download checkpoints from Huggingface.
| Task | Hub repo |
| --- | --- |
| Toxicity | [`hayleyson/laser-edit-toxicity-energy`](https://huggingface.co/hayleyson/laser-edit-toxicity-energy) |
| NLI (contradiction avoidance) | [`hayleyson/laser-edit-nli-energy`](https://huggingface.co/hayleyson/laser-edit-nli-energy) |

Example:

```bash
# Suggested layout under the repo root
mkdir -p checkpoints/energy

huggingface-cli download hayleyson/laser-edit-toxicity-energy \
  --local-dir checkpoints/energy/toxicity

huggingface-cli download hayleyson/laser-edit-nli-energy \
  --local-dir checkpoints/energy/nli
```

Then update entrypoints to point to the local directories:

| Method family | What to change |
| --- | --- |
| **LaSEr & LLM Edit** (`laser_llm/`, and LLM-edit stages that evaluate with an EBM) | Set `PRETRAINED_MODEL_PATH` / `PRETRAINED_MODEL_PATH_TOX` / `PRETRAINED_MODEL_PATH_NLI` to the download dirs (e.g. `checkpoints/energy/toxicity`, `checkpoints/energy/nli`). |
| **LaSEr & EBM Edit** (`laser_ebm/toxicity.sh`, `nli.sh`, `multi.sh`) | In `--model_paths` and `--tokenizer_paths`, keep the base LM (e.g. Qwen) first; set the **energy** path(s) to the same local dirs. For `multi`, pass NLI then toxicity (order must match the task). |
| **Plain / self-locate LLM** scripts that pass `--pretrained_model_path` | Same as above when an energy checkpoint is required for evaluation or locate. |

### Set-consistency (set-LConVQA / set-SNLI)

For **set-consistency enforcement** tasks, download the trained set-consistency energy weights from the official [SC_Energy_public](https://github.com/radishtiger/SC_Energy_public) release (Google Drive link under **Model Weights** in that README).

Update `model_path` in LaSEr’s set-consistency configs:

   - `laser_edit/set_consistency_energy/params_set_lconvqa.yaml` (set-LConVQA)
   - `laser_edit/set_consistency_energy/params_set_nli.yaml` (set-SNLI)

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

---

## Evaluation

Task-specific evaluation scripts live under:

```text
laser_edit/evaluation/scripts/
```

Point each script’s `GENERATIONS_FILE_PATH` (and `SOURCE_FILE_PATH` if needed) at the run you want to score, then submit with `sbatch`.

For **toxicity** and **nli** LLM-edit outputs, evaluation applies metric-specific postprocessing first (fluency for both; also NLI formatting for nli) before scoring those metrics. EBM outputs are evaluated directly.

| Task | Edit type | Script |
| --- | --- | --- |
| toxicity | EBM | `toxicity_ebm.sh` |
| toxicity | LLM | `toxicity_llm.sh` |
| nli | EBM | `nli_ebm.sh` |
| nli | LLM | `nli_llm.sh` |
| multi | EBM or LLM | `multi.sh` |
| set-LConVQA | EBM | `set_lconvqa_ebm.sh` |
| set-LConVQA | LLM | `set_lconvqa_llm.sh` |
| set-NLI (set-SNLI) | EBM | `set_nli_ebm.sh` |
| set-NLI (set-SNLI) | LLM | `set_nli_llm.sh` |

Example:

```bash
# Edit GENERATIONS_FILE_PATH inside the script first
sbatch laser_edit/evaluation/scripts/nli_llm.sh
```