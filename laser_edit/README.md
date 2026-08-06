# LaSEr-Edit

**[Paper (arXiv)](https://arxiv.org/abs/2407.00740)** · Localized Span-level Error Editing with Energy-based Localization

Official code for **[LaSEr-Edit](https://arxiv.org/abs/2407.00740)** (Son et al., 2026), a constraint-satisfying text revision framework that (1) localizes constraint-violating spans with lightweight energy-based models (EBMs) and (2) edits only those spans.

We provide two main editing variants from the paper:

- **LaSEr-LLM Edit** — EBM localization + LLM span editing  
- **LaSEr-EBM Edit** — EBM localization + EBM-guided span editing (candidate ranking)

Baselines used in the paper (plain / self-locate LLM editing) are also included.

---

## Contents

- [Tasks](#tasks)
- [Methods](#methods)
- [Setup](#setup)
- [Energy model checkpoints](#energy-model-checkpoints)
- [Running experiments](#running-experiments)
- [Evaluation](#evaluation)

All commands assume the **repository root** (parent of `laser_edit/`) with `PYTHONPATH=.`:

```bash
cd /path/to/mucoco
export PYTHONPATH=.
```

Preferred experiment entrypoints:

```text
laser_edit/_bash_scripts/entrypoints/
```

---

## Tasks

| Task (paper) | Script nickname | Constraint |
| --- | --- | --- |
| Toxicity avoidance | `toxicity` | Non-toxicity |
| Contradiction avoidance | `nli` | Single-pair logical consistency (NLI) |
| Set-consistency Enforcement (LConVQA) | `set_lconvqa` | Consistency over a set of QA pairs |
| Set-consistency Enforcement (Set-SNLI) | `set_nli` | Consistency over a set of sentences |
| Joint toxicity and contradiction avoidance | `multi` | Joint non-toxicity + pairwise consistency |

---

## Methods

| Method | Localization | Editing |
| --- | --- | --- |
| Plain LLM Edit | N/A | LLM |
| Self-locate & LLM Edit | same LLM proposes spans | LLM |
| Self-parallel-locate & LLM Edit | separate LLM locates per constraint, then union | LLM |
| **LaSEr-LLM Edit** | EBM (LaSEr) | LLM |
| **LaSEr-EBM Edit** | EBM (LaSEr) | EBM-guided decoding |
| **LaSEr-EBM Edit + LLM smoothing** | EBM (LaSEr) | EBM-guided decoding → LLM smoothing for fluency |

---

## Setup

Create **two** conda environments from `laser_edit/`:

```bash
# 1) General / EBM stack (no vLLM)
bash create_env_py3.11_cuda12.4.sh

# 2) LLM paths that need vLLM (CUDA 13.0)
bash create_env_py3.12_cuda13.0_vllm.sh
```

Use the vLLM environment (`loc-edit-pro6000-vllm`) when running LLM edit scripts with `--use_vllm`.

---

## Energy model checkpoints

LaSEr localization and LaSEr-EBM Edit require local EBM checkpoints.

### Toxicity and NLI

| Task | Hugging Face repo |
| --- | --- |
| Toxicity | [`hayleyson/laser-edit-toxicity-energy`](https://huggingface.co/hayleyson/laser-edit-toxicity-energy) |
| NLI | [`hayleyson/laser-edit-nli-energy`](https://huggingface.co/hayleyson/laser-edit-nli-energy) |

```bash
mkdir -p checkpoints/energy

huggingface-cli download hayleyson/laser-edit-toxicity-energy \
  --local-dir checkpoints/energy/toxicity

huggingface-cli download hayleyson/laser-edit-nli-energy \
  --local-dir checkpoints/energy/nli
```

Point entrypoints at these directories:

| Method family | Config to update |
| --- | --- |
| LaSEr-LLM Edit (`laser_llm/`, etc.) | `PRETRAINED_MODEL_PATH` / `_TOX` / `_NLI` |
| LaSEr-EBM Edit (`laser_ebm/`) | energy paths in `--model_paths` / `--tokenizer_paths` (base LM first; for `multi`, NLI then toxicity) |
| Plain / self-locate scripts that score with an EBM | `--pretrained_model_path` as needed |

### Set-consistency (Set-LConVQA / set-SNLI)

Download set-consistency energy weights from [SC_Energy_public](https://github.com/radishtiger/SC_Energy_public) (Google Drive link under **Model Weights**), then set `model_path` in:

- `laser_edit/set_consistency_energy/params_set_lconvqa.yaml`
- `laser_edit/set_consistency_energy/params_set_nli.yaml`

---

## Running experiments

One script (or short pipeline) per method × task. Submit with `sbatch` after updating paths / job-specific filenames as needed.

| Method | Task | Entrypoint |
| --- | --- | --- |
| Plain LLM Edit | toxicity | `sbatch laser_edit/_bash_scripts/entrypoints/plain_llm/toxicity.sh` |
| Plain LLM Edit | nli | `…/plain_llm/nli.sh` |
| Plain LLM Edit | Set-LConVQA | `…/plain_llm/set_lconvqa.sh` |
| Plain LLM Edit | Set-SNLI | `…/plain_llm/set_nli.sh` |
| Plain LLM Edit | multi | `…/plain_llm/multi.sh` |
| Self-locate & LLM Edit | toxicity | `…/self_locate_llm/toxicity_{locate,postprocess,edit}.sh` |
| Self-locate & LLM Edit | nli | `…/self_locate_llm/nli_{locate,postprocess,edit}.sh` |
| Self-locate & LLM Edit | Set-LConVQA | `…/self_locate_llm/set_lconvqa_{locate,edit}.sh` |
| Self-locate & LLM Edit | Set-SNLI | `…/self_locate_llm/set_nli_{locate,edit}.sh` |
| Self-locate & LLM Edit | multi | `…/self_locate_llm/multi_{locate,postprocess,edit}.sh` |
| Self-parallel-locate & LLM Edit | multi | `…/self_parallel_locate_llm/multi_{locate,postprocess,edit}.sh` |
| LaSEr-LLM Edit | toxicity | `…/laser_llm/toxicity.sh` |
| LaSEr-LLM Edit | nli | `…/laser_llm/nli.sh` |
| LaSEr-LLM Edit | Set-LConVQA | `…/laser_llm/set_lconvqa_{locate,edit}.sh` |
| LaSEr-LLM Edit | Set-SNLI | `…/laser_llm/set_nli_{locate,edit}.sh` |
| LaSEr-LLM Edit | multi | `…/laser_llm/multi.sh` |
| LaSEr-EBM Edit | toxicity | `…/laser_ebm/toxicity.sh` |
| LaSEr-EBM Edit | nli | `…/laser_ebm/nli.sh` |
| LaSEr-EBM Edit | Set-LConVQA | `…/laser_ebm/set_lconvqa.sh` |
| LaSEr-EBM Edit | Set-SNLI | `…/laser_ebm/set_nli.sh` |
| LaSEr-EBM Edit | multi | `…/laser_ebm/multi.sh` |
| + LLM smoothing | all tasks | `…/laser_ebm_llm_smoothing/<task>.sh` (set `INPUT_PATH` first) |

Multi-step pipelines: run **locate → postprocess → edit** in order. After locate, update result filenames in later scripts (they include timestamps / job ids).

Default hyperparameters follow the paper (see Tables 13–14 in the [arXiv PDF](https://arxiv.org/pdf/2407.00740)).

---

## Evaluation

Scripts: `laser_edit/evaluation/scripts/`.

Point `GENERATIONS_FILE_PATH` (and `SOURCE_FILE_PATH` if needed) at the run to score, then `sbatch`.

For **toxicity** and **nli** LLM-edit outputs, metric-specific postprocessing is applied before fluency (and, for nli, the NLI metric). EBM `outputs.txt` files are evaluated directly.

| Task | Edit type | Script |
| --- | --- | --- |
| toxicity | EBM | `toxicity_ebm.sh` |
| toxicity | LLM | `toxicity_llm.sh` |
| nli | EBM | `nli_ebm.sh` |
| nli | LLM | `nli_llm.sh` |
| multi | EBM or LLM | `multi.sh` |
| Set-LConVQA | EBM | `set_lconvqa_ebm.sh` |
| Set-LConVQA | LLM | `set_lconvqa_llm.sh` |
| Set-SNLI | EBM | `set_nli_ebm.sh` |
| Set-SNLI | LLM | `set_nli_llm.sh` |

```bash
# Edit GENERATIONS_FILE_PATH inside the script first
sbatch laser_edit/evaluation/scripts/nli_llm.sh
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{son2024laser,
  title={LaSEr-Edit: Localized Span-level Error Editing with Energy-based Localization},
  author={Son, Hye Ryung and Eom, Saehee and Song, Mooho and Lee, Jay-Yoon},
  journal={arXiv preprint arXiv:2407.00740},
  year={2024}
}
```