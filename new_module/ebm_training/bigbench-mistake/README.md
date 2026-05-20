# Mistake-Finding Classifier Training

This directory contains code for training mistake-finding classifiers as described in the paper. The classifiers are fine-tuned from PaLM 2 Otter to predict whether a CoT (Chain-of-Thought) step is correct or incorrect.

## Overview

The training procedure:
1. Fine-tunes PaLM 2 Otter on 4 out of 5 tasks, holding out one task for evaluation
2. This is done for each of the 5 tasks (5 different models total)
3. Each model is trained with specific hyperparameters and for a different number of steps

## Tasks

The 5 tasks are:
- `word_sorting`
- `tracking_shuffled_objects`
- `logical_deduction`
- `multi_step_arithmetic`
- `dyck_languages`

## Training Parameters

- **Max steps**: 20,000 (but each model stops at different steps based on validation)
- **Batch size**: 32
- **Learning rate**: 1e-5
- **LR schedule**: Linear warmup + cosine decay
- **Warmup steps**: 500

### Training Steps per Task (from Table 8)

| Held-out Task | Training Steps |
|--------------|----------------|
| Word sorting | 6,800 |
| Tracking shuffled objects | 8,000 |
| Logical deduction | 9,000 |
| Multi-step arithmetic | 10,000 |
| Dyck languages | 10,000 |

## Data Format

The training script expects data files in JSONL format, one file per task:
- `{data_dir}/word_sorting.jsonl`
- `{data_dir}/tracking_shuffled_objects.jsonl`
- `{data_dir}/logical_deduction.jsonl`
- `{data_dir}/multi_step_arithmetic.jsonl`
- `{data_dir}/dyck_languages.jsonl`

Each line in the JSONL file should be a JSON object with:
- `input`: The input/problem text
- `steps`: A list of strings, where each string is a CoT step (already parsed)
- `mistake_index`: The index (0-based) of the first step where a mistake occurred. If `null`, there are no mistakes (all steps are correct).

Example:
```json
{
  "input": "Sort the words: cat, apple, dog",
  "steps": [
    "First, I'll identify the words: cat, apple, dog",
    "Now I'll sort them alphabetically: apple, cat, dog",
    "The sorted order is: apple, cat, dog"
  ],
  "mistake_index": null
}
```

For labeling:
- Steps before `mistake_index` are labeled as correct (1)
- Steps at and after `mistake_index` are labeled as incorrect (0)
- If `mistake_index` is `null`, all steps are labeled as correct (1)

## Usage

### Basic Usage

Train a classifier for a specific held-out task:
```bash
python train_classifier.py \
  --data_dir /path/to/data \
  --held_out_task word_sorting \
  --output_base_dir ./checkpoints
```

### Train All Models

Train classifiers for all 5 tasks:
```bash
python train_classifier.py \
  --data_dir /path/to/data \
  --output_base_dir ./checkpoints
```

### Custom Model

Use a different base model:
```bash
python train_classifier.py \
  --model_name google/palm-2-otter \
  --data_dir /path/to/data \
  --held_out_task logical_deduction \
  --output_base_dir ./checkpoints
```

### Custom Hyperparameters

Override default hyperparameters:
```bash
python train_classifier.py \
  --data_dir /path/to/data \
  --held_out_task multi_step_arithmetic \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --output_base_dir ./checkpoints
```

## Output

For each held-out task, the script creates:
- `{output_base_dir}/classifier_heldout_{task_name}/`
  - `pytorch_model.bin`: The trained model weights
  - `config.json`: Model configuration
  - `tokenizer_config.json`: Tokenizer configuration
  - `eval_results.json`: Final evaluation metrics
  - `checkpoint-{step}/`: Intermediate checkpoints (best model is loaded at end)

## Implementation Details

### Data Processing

1. **CoT Step Loading**: The script loads steps directly from the `steps` list in the JSONL files (no parsing needed)
2. **Label Assignment**: Labels are assigned based on `mistake_index`:
   - Steps before `mistake_index` are labeled as correct (1)
   - Steps at and after `mistake_index` are labeled as incorrect (0)
   - If `mistake_index` is `null`, all steps are labeled as correct (1)
3. **Input Formatting**: Each training example includes:
   - Task name
   - Input/problem text
   - Previous steps (context)
   - Current step to evaluate

### Training Process

1. **Hold-out Split**: For each task, 4 tasks are used for training, 1 task is held out for validation
2. **Early Stopping**: Training stops early if validation F1 doesn't improve for 3 evaluation steps
3. **Checkpoint Selection**: The best checkpoint (based on validation F1) is selected and saved
4. **Step Limiting**: Each model stops at the specified number of steps from Table 8

## Notes

- The script assumes the base model is available via HuggingFace or locally. Adjust `--model_name` if using a different model or local path.
- If your data format differs, you may need to modify the `load_task_data` and `process_cot_steps` functions.
- The script uses FP16 training if CUDA is available for faster training.
- The data format expects `steps` to be a list of strings and `mistake_index` to be an integer or `null`.

