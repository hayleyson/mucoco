"""
Training script for energy-based mistake-finding models as described in the paper.

Fine-tunes PaLM 2 Otter on 4 out of 5 tasks, holding out one task for evaluation.
This is done for each of the 5 tasks.

The model assigns scalar energy values to CoT prefixes, with lower energy
indicating more plausible/correct reasoning.

Training objective consists of four margin-based ranking losses:
1. Positive-negative contrastive loss
2. Length-based contrast for negative prefixes
3. Length-based contrast for positive prefixes
4. Energy-jump loss at mistake location

Training parameters:
- 20k steps max
- Batch size: 32
- Learning rate: 1e-5 with linear ramp and cosine decay
- Select checkpoint with best validation results
- Different models stop at different steps (see TRAINING_STEPS dict)
"""

import os
import json
import re
import argparse
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# Task names
TASKS = [
    "word_sorting",
    "tracking_shuffled_objects",
    "logical_deduction",
    "multistep_arithmetic",
    "dyck_languages",
]

# Number of training steps for each held-out task (from Table 8)
TRAINING_STEPS = {
    "word_sorting": 6800,
    "tracking_shuffled_objects": 8000,
    "logical_deduction": 9000,
    "multistep_arithmetic": 10000,
    "dyck_languages": 10000,
}

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MAX_STEPS = 20000
WARMUP_STEPS = 500  # For linear ramp
MINIMUM_BATCH_SIZE = 4

# Energy model loss hyperparameters
# Margins
DELTA_1 = 1.0  # Positive-negative contrast margin (must be > 0)
DELTA_2 = 0.0  # Negative length-based contrast margin (>= 0, can be 0 for monotonicity)
DELTA_3 = 0.0  # Positive length-based contrast margin (>= 0, can be 0 for monotonicity)
DELTA_4 = 1.0  # Energy-jump margin (must be > 0)

# Loss weights
LAMBDA_1 = 1.0  # Weight for positive-negative contrast
LAMBDA_2 = 1.0  # Weight for negative length-based contrast
LAMBDA_3 = 1.0  # Weight for positive length-based contrast
LAMBDA_4 = 1.0  # Weight for energy-jump loss


@dataclass
class CoTPrefixExample:
    """Represents a single CoT prefix for energy model training."""
    task: str
    input_text: str
    prefix_text: str  # The prefix s_{<=t} as formatted text
    prefix_length: int  # Length t of the prefix (number of steps)
    is_positive: bool  # True if prefix is correct, False if incorrect
    trace_id: str  # Unique identifier for the trace (to group prefixes)
    mistake_index: Optional[int]  # Index of first mistake in the trace (None if no mistake)


def process_cot_trace(
    task: str,
    input_text: str,
    steps: List[str],
    mistake_index: Optional[int],
    trace_id: str
) -> List[CoTPrefixExample]:
    """
    Process a CoT trace into prefix examples for energy model training.
    
    Args:
        task: Name of the task
        input_text: The input/problem text
        steps: List of step strings (already parsed)
        mistake_index: Index of the first step where a mistake occurred (None if no mistake)
        trace_id: Unique identifier for this trace
    
    Returns:
        List of CoTPrefixExample objects, one for each prefix length
    """
    if not steps:
        return []
    
    examples = []
    
    # Build prefixes incrementally
    prefix_steps = []
    for i, step_text in enumerate(steps):
        prefix_steps.append(step_text)
        
        # Determine if this prefix is positive (correct) or negative (incorrect)
        # Prefixes before mistake_index are positive, at/after are negative
        if mistake_index is None:
            # No mistake, all prefixes are positive
            is_positive = True
        elif i < mistake_index:
            # Prefix is before the first mistake, so it's positive
            is_positive = True
        else:
            # Prefix is at or after the first mistake, so it's negative
            is_positive = False
        
        # Format the prefix text
        prefix_parts = []
        for j, step in enumerate(prefix_steps):
            prefix_parts.append(f"Step {j+1}: {step}")
        prefix_text = "\n".join(prefix_parts)
        
        # Create the example
        example = CoTPrefixExample(
            task=task,
            input_text=input_text,
            prefix_text=prefix_text,
            prefix_length=i + 1,  # 1-indexed length
            is_positive=is_positive,
            trace_id=trace_id,
            mistake_index=mistake_index
        )
        examples.append(example)
    
    return examples


def format_input_for_energy_model(example: CoTPrefixExample) -> str:
    """
    Format an example into the input text for the energy model.
    
    The energy model should predict the energy of a prefix, given:
    - The task
    - The input/problem
    - The prefix (all steps up to current)
    """
    # Format: Task: <task>\nInput: <input>\n<prefix_text>
    parts = [f"Task: {example.task}"]
    parts.append(f"Input: {example.input_text}")
    parts.append(example.prefix_text)
    
    return "\n".join(parts)


class EnergyModelDataset(Dataset):
    """Dataset for energy-based mistake-finding model."""
    
    def __init__(
        self,
        examples: List[CoTPrefixExample],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Group examples by trace_id for loss computation
        self.trace_groups = defaultdict(list)
        for idx, example in enumerate(examples):
            self.trace_groups[example.trace_id].append(idx)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = format_input_for_energy_model(example)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(1 if example.is_positive else 0, dtype=torch.long),  # For Trainer compatibility
            'is_positive': torch.tensor(1 if example.is_positive else 0, dtype=torch.long),  # For loss computation
            'prefix_length': torch.tensor(example.prefix_length, dtype=torch.long),
            'mistake_index': torch.tensor(
                example.mistake_index if example.mistake_index is not None else -1,
                dtype=torch.long
            ),
            'example_idx': torch.tensor(idx, dtype=torch.long)
        }
    
    def get_trace_indices(self, trace_id: str) -> List[int]:
        """Get all example indices for a given trace."""
        return self.trace_groups.get(trace_id, [])


def load_task_data(task_name: str, data_path: Optional[str] = None) -> List[Dict]:
    """
    Load data for a specific task.
    
    Args:
        task_name: Name of the task
        data_path: Path to data file (JSONL format)
    
    Returns:
        List of examples with 'input', 'steps', 'mistake_index', and 'task' fields
    """
    if data_path is None:
        raise ValueError(f"Please provide data_path for task {task_name}")
    
    # Load from local JSONL file
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                # Add task name to the example
                example['task'] = task_name
                examples.append(example)
    
    return examples


def create_train_val_split(
    all_tasks: List[str],
    held_out_task: str,
    data_dir: str,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[CoTPrefixExample], List[CoTPrefixExample]]:
    """
    Create train/val split by randomly splitting training tasks.
    The held_out_task is excluded and reserved for testing only.
    
    Args:
        all_tasks: List of all task names
        held_out_task: Task to hold out for testing (excluded from train/val)
        data_dir: Directory containing task data files
        val_ratio: Ratio of examples to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_examples, val_examples)
    """
    all_examples = []
    trace_counter = 0
    
    # Collect all examples from tasks other than the held-out task
    for task in all_tasks:
        # Skip the held-out task (reserved for testing)
        if task == held_out_task:
            continue
        
        # Load data for this task
        data_path = os.path.join(data_dir, f"{task}.jsonl")
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found for task {task}: {data_path}")
            continue
        
        task_data = load_task_data(task, data_path)
        
        for example_data in task_data:
            input_text = example_data.get('input', '')
            steps = example_data.get('steps', [])
            mistake_index = example_data.get('mistake_index', None)
            
            # Handle None mistake_index (no mistake, all steps correct)
            if mistake_index is not None and not isinstance(mistake_index, int):
                # Skip invalid examples
                continue
            
            if not steps:
                continue
            
            # Generate unique trace ID
            trace_id = f"{task}_{trace_counter}"
            trace_counter += 1
            
            # Process trace into prefix examples
            prefix_examples = process_cot_trace(
                task=task,
                input_text=input_text,
                steps=steps,
                mistake_index=mistake_index,
                trace_id=trace_id
            )
            
            all_examples.extend(prefix_examples)
    
    # Randomly shuffle and split into train and validation
    random.seed(random_seed)
    random.shuffle(all_examples)
    
    val_size = int(len(all_examples) * val_ratio)
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]
    
    return train_examples, val_examples



def compute_energy_losses(
    energies: torch.Tensor,
    is_positive: torch.Tensor,
    prefix_lengths: torch.Tensor,
    mistake_indices: torch.Tensor,
    trace_ids: List[str],
    delta_1: float = DELTA_1,
    delta_2: float = DELTA_2,
    delta_3: float = DELTA_3,
    delta_4: float = DELTA_4,
    lambda_1: float = LAMBDA_1,
    lambda_2: float = LAMBDA_2,
    lambda_3: float = LAMBDA_3,
    lambda_4: float = LAMBDA_4,
    pos_len_no_mistake_only: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the four energy-based losses.
    
    Args:
        energies: Tensor of shape (batch_size,) containing energy values
        is_positive: Tensor of shape (batch_size,) with 1 for positive, 0 for negative
        prefix_lengths: Tensor of shape (batch_size,) with prefix lengths
        mistake_indices: Tensor of shape (batch_size,) with mistake indices (-1 if no mistake)
        trace_ids: List of trace IDs for each example in the batch
        delta_1, delta_2, delta_3, delta_4: Margin hyperparameters
        lambda_1, lambda_2, lambda_3, lambda_4: Loss weights
    
    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
    """
    device = energies.device
    batch_size = energies.size(0)
    
    # Group examples by trace_id
    trace_groups = defaultdict(list)
    for i, trace_id in enumerate(trace_ids):
        trace_groups[trace_id].append(i)
    
    # Initialize loss accumulators
    loss_pos_neg = torch.tensor(0.0, device=device)
    loss_neg_len = torch.tensor(0.0, device=device)
    loss_pos_len = torch.tensor(0.0, device=device)
    loss_jump = torch.tensor(0.0, device=device)
    
    count_pos_neg = 0
    count_neg_len = 0
    count_pos_len = 0
    count_jump = 0
    
    # 1. Positive-negative contrast (across all traces)
    # Collect all positive and negative examples from the entire batch
    pos_mask_all = is_positive.bool()
    neg_mask_all = ~pos_mask_all
    
    all_pos_energies = energies[pos_mask_all]
    all_neg_energies = energies[neg_mask_all]
    
    if len(all_pos_energies) > 0 and len(all_neg_energies) > 0:
        # For each positive-negative pair across all traces
        for pos_e in all_pos_energies:
            for neg_e in all_neg_energies:
                margin_loss = torch.clamp(pos_e + delta_1 - neg_e, min=0.0)
                loss_pos_neg += margin_loss
                count_pos_neg += 1
    
    # Process each trace for losses 2, 3, and 4
    for trace_id, indices in trace_groups.items():
        if len(indices) < 2:
            continue  # Need at least 2 prefixes to compute length-based losses
        
        trace_energies = energies[indices]
        trace_is_positive = is_positive[indices]
        trace_lengths = prefix_lengths[indices]
        trace_mistake_idx = mistake_indices[indices[0]].item()  # Same for all in trace
        
        # Separate positive and negative prefixes
        pos_mask = trace_is_positive.bool()
        neg_mask = ~pos_mask
        
        pos_indices = [indices[i] for i in range(len(indices)) if pos_mask[i]]
        neg_indices = [indices[i] for i in range(len(indices)) if neg_mask[i]]
        
        pos_energies = trace_energies[pos_mask]
        neg_energies = trace_energies[neg_mask]
        pos_lengths = trace_lengths[pos_mask]
        neg_lengths = trace_lengths[neg_mask]
        
        # 2. Length-based negative-negative contrast
        if len(neg_energies) >= 2:
            # Sort by length
            neg_sorted = sorted(zip(neg_lengths, neg_energies), key=lambda x: x[0])
            for i in range(len(neg_sorted) - 1):
                len1, e1 = neg_sorted[i]
                len2, e2 = neg_sorted[i + 1]
                if len1 < len2:  # Ensure len1 < len2
                    margin_loss = torch.clamp(e1 + delta_2 - e2, min=0.0)
                    loss_neg_len += margin_loss
                    count_neg_len += 1
        
        # 3. Length-based positive-positive contrast
        if len(pos_energies) >= 2:
            # Filter positive examples if pos_len_no_mistake_only is enabled
            if pos_len_no_mistake_only:
                # Only include positive examples from traces with no mistake (mistake_index == -1)
                # All examples in a trace have the same mistake_index, so check trace_mistake_idx
                if trace_mistake_idx == -1:
                    pos_energies_filtered = pos_energies
                    pos_lengths_filtered = pos_lengths
                else:
                    # Skip this trace if it has a mistake
                    pos_energies_filtered = torch.tensor([], device=device, dtype=pos_energies.dtype)
                    pos_lengths_filtered = torch.tensor([], device=device, dtype=pos_lengths.dtype)
            else:
                pos_energies_filtered = pos_energies
                pos_lengths_filtered = pos_lengths
            
            if len(pos_energies_filtered) >= 2:
                # Sort by length
                pos_sorted = sorted(zip(pos_lengths_filtered, pos_energies_filtered), key=lambda x: x[0])
                for i in range(len(pos_sorted) - 1):
                    len1, e1 = pos_sorted[i]
                    len2, e2 = pos_sorted[i + 1]
                    if len1 < len2:  # Ensure len1 < len2
                        # For positive prefixes, longer should have lower energy
                        margin_loss = torch.clamp(e2 + delta_3 - e1, min=0.0)
                        loss_pos_len += margin_loss
                        count_pos_len += 1
        
        # 4. Energy-jump loss
        # Only compute for traces with mistakes (trace_mistake_idx != -1)
        if trace_mistake_idx != -1 and len(trace_energies) > 1:
            # Compute energy increments (discrete derivatives)
            # Delta_t = e(x, s_{<=t}) - e(x, s_{<=t-1})
            # For t=0, we use e(x, s_{<=0}) = 0
            increments = []
            prev_energy = torch.tensor(0.0, device=device)  # e(x, s_{<=0}) = 0
            
            sorted_by_length = sorted(
                indices,
                key=lambda x: prefix_lengths[x].item()
            )
            
            mistake_step_idx = None
            for idx_in_sorted, orig_idx in enumerate(sorted_by_length):
                length = prefix_lengths[orig_idx].item()
                energy = energies[orig_idx]
                increment = energy - prev_energy
                increments.append((length, increment, orig_idx))
                
                if length == trace_mistake_idx + 1:  # mistake_index is 0-indexed, length is 1-indexed
                    mistake_step_idx = idx_in_sorted
                
                prev_energy = energy
            
            if mistake_step_idx is not None and len(increments) > 1:
                mistake_increment = increments[mistake_step_idx][1]
                for length, increment, _ in increments:
                    if length != trace_mistake_idx + 1:  # Skip the mistake step itself
                        margin_loss = torch.clamp(increment + delta_4 - mistake_increment, min=0.0)
                        loss_jump += margin_loss
                        count_jump += 1
    
    # Normalize by counts (average)
    if count_pos_neg > 0:
        loss_pos_neg = loss_pos_neg / count_pos_neg
    if count_neg_len > 0:
        loss_neg_len = loss_neg_len / count_neg_len
    if count_pos_len > 0:
        loss_pos_len = loss_pos_len / count_pos_len
    if count_jump > 0:
        loss_jump = loss_jump / count_jump
    
    # Weighted sum
    total_loss = (
        lambda_1 * loss_pos_neg +
        lambda_2 * loss_neg_len +
        lambda_3 * loss_pos_len +
        lambda_4 * loss_jump
    )
    
    loss_dict = {
        'loss_pos_neg': loss_pos_neg,
        'loss_neg_len': loss_neg_len,
        'loss_pos_len': loss_pos_len,
        'loss_jump': loss_jump,
        'count_pos_neg': count_pos_neg,
        'count_neg_len': count_neg_len,
        'count_pos_len': count_pos_len,
        'count_jump': count_jump,
    }
    
    return total_loss, loss_dict


class EnergyModel(nn.Module):
    """Energy model that outputs scalar energy values."""
    
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        # Energy head: single scalar output
        self.energy_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Get outputs from the base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Get the last hidden state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        
        # For causal LMs, we use the last non-padding token's hidden state
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            sequence_lengths = torch.clamp(sequence_lengths, 0, last_hidden_state.size(1) - 1)
            pooled_output = last_hidden_state[range(len(sequence_lengths)), sequence_lengths]
        else:
            pooled_output = last_hidden_state[:, -1, :]
        
        # Compute energy (scalar)
        energy = self.energy_head(pooled_output).squeeze(-1)  # (batch_size,)
        
        return energy
    
    def save_pretrained(self, save_directory):
        """Save the energy model."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        # Save base model
        self.base_model.save_pretrained(save_directory)
        # Save energy head
        torch.save(self.energy_head.state_dict(), os.path.join(save_directory, "energy_head.pt"))
        # Save config with energy model info
        config = self.base_model.config
        if hasattr(config, 'save_pretrained'):
            config.save_pretrained(save_directory)


class EnergyModelTrainer(Trainer):
    """Custom Trainer for energy-based models with margin losses."""
    
    def __init__(
        self,
        delta_1: float = DELTA_1,
        delta_2: float = DELTA_2,
        delta_3: float = DELTA_3,
        delta_4: float = DELTA_4,
        lambda_1: float = LAMBDA_1,
        lambda_2: float = LAMBDA_2,
        lambda_3: float = LAMBDA_3,
        lambda_4: float = LAMBDA_4,
        pos_len_no_mistake_only: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.delta_3 = delta_3
        self.delta_4 = delta_4
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.pos_len_no_mistake_only = pos_len_no_mistake_only
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the energy-based margin losses.
        """
        # Get energy predictions
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        
        energies = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get labels and metadata
        is_positive = inputs.get('is_positive')
        prefix_lengths = inputs.get('prefix_length')
        mistake_indices = inputs.get('mistake_index')
        
        # Get trace_ids from example_idx (since trace_id strings can't be batched as tensors)
        example_indices = inputs.get('example_idx')
        if example_indices is not None:
            # Get the dataset - try train_dataset first, then eval_dataset
            dataset = None
            if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                # Check if indices are valid for train dataset
                max_idx = max(idx.item() for idx in example_indices)
                if max_idx < len(self.train_dataset.examples):
                    dataset = self.train_dataset
            
            if dataset is None and hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
                # Check if indices are valid for eval dataset
                max_idx = max(idx.item() for idx in example_indices)
                if max_idx < len(self.eval_dataset.examples):
                    dataset = self.eval_dataset
            
            if dataset is not None:
                trace_ids = [
                    dataset.examples[idx.item()].trace_id
                    for idx in example_indices
                ]
            else:
                # Fallback: create dummy trace IDs
                trace_ids = [f"trace_{i}" for i in range(energies.size(0))]
        else:
            # Fallback: create dummy trace IDs (one per example)
            trace_ids = [f"trace_{i}" for i in range(energies.size(0))]
        
        # Compute losses
        total_loss, loss_dict = compute_energy_losses(
            energies=energies,
            is_positive=is_positive,
            prefix_lengths=prefix_lengths,
            mistake_indices=mistake_indices,
            trace_ids=trace_ids,
            delta_1=self.delta_1,
            delta_2=self.delta_2,
            delta_3=self.delta_3,
            delta_4=self.delta_4,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            lambda_3=self.lambda_3,
            lambda_4=self.lambda_4,
            pos_len_no_mistake_only=self.pos_len_no_mistake_only,
        )
        
        # Store loss components for logging
        if not hasattr(self, '_loss_components'):
            self._loss_components = {}
        self._loss_components.update(loss_dict)
        
        if return_outputs:
            return total_loss, {'energies': energies, **loss_dict}
        return total_loss
    
    def log(self, logs):
        """Override log to include loss components."""
        # Add loss components to logs
        if hasattr(self, '_loss_components'):
            for key, value in self._loss_components.items():
                if isinstance(value, torch.Tensor):
                    logs[f'train/{key}'] = value.item()
                else:
                    logs[f'train/{key}'] = value
        super().log(logs)


def _tune_threshold_on_traces(trace_predictions, trace_labels):
    """
    Tune threshold for trace classification by finding the threshold that maximizes accuracy.
    
    Args:
        trace_predictions: dict mapping trace_id -> (max_energy, has_mistake)
        trace_labels: dict mapping trace_id -> bool (has_mistake)
    
    Returns:
        Best threshold value
    """
    if len(trace_predictions) == 0:
        return 0.0
    
    trace_energies = [trace_predictions[tid][0] for tid in trace_predictions]
    min_energy = min(trace_energies)
    max_energy = max(trace_energies)
    
    # Try different thresholds and find the one with best accuracy
    best_threshold = np.median(trace_energies)
    best_accuracy = 0.0
    
    # Try percentiles and some specific values
    candidate_thresholds = np.percentile(trace_energies, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    candidate_thresholds = np.concatenate([candidate_thresholds, [min_energy, max_energy, np.median(trace_energies)]])
    
    for threshold in candidate_thresholds:
        correct = 0
        total = 0
        for trace_id in trace_predictions:
            pred_has_mistake = trace_predictions[trace_id][0] > threshold
            true_has_mistake = trace_labels[trace_id]
            if pred_has_mistake == true_has_mistake:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold


# Global variable to store tuned threshold from callback
_tuned_threshold_global = None

def compute_energy_metrics(eval_pred, eval_dataset=None, threshold_tuning_dataset=None, tuned_threshold=None):
    """
    Compute metrics for energy model evaluation.
    
    The model is trained to:
    1. Make positive prefixes have lower energy than negative ones
    2. Make energy jump at mistake locations
    3. Enforce length-based monotonicity
    
    We evaluate:
    1. Energy separation: How well does the model separate positive vs negative prefixes
    2. Trace-level classification: Can the model identify traces with mistakes?
    3. Energy statistics: Mean energies and gaps
    
    Args:
        eval_pred: Predictions and labels from the evaluation
        eval_dataset: Dataset used for computing metrics (metrics reporting set)
        threshold_tuning_dataset: Dataset used for tuning the threshold (threshold tuning set)
    """
    # Handle both tuple and EvalPrediction formats
    if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    
    # predictions are energies, labels are is_positive (1 for positive, 0 for negative)
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    energies = predictions.flatten() if predictions.ndim > 1 else predictions
    is_positive = labels.flatten() if labels.ndim > 1 else labels
    
    # Compute average energy for positive vs negative
    pos_energies = energies[is_positive == 1]
    neg_energies = energies[is_positive == 0]
    
    # Energy separation metrics (what the model is actually trained for)
    mean_energy_pos = float(np.mean(pos_energies)) if len(pos_energies) > 0 else 0.0
    mean_energy_neg = float(np.mean(neg_energies)) if len(neg_energies) > 0 else 0.0
    energy_gap = mean_energy_neg - mean_energy_pos
    
    # Separation quality: percentage of positive examples with energy < median of all energies
    # This measures how well the model separates positive from negative
    median_energy = np.median(energies)
    pos_below_median = np.sum(pos_energies < median_energy) if len(pos_energies) > 0 else 0
    pos_separation_rate = pos_below_median / len(pos_energies) if len(pos_energies) > 0 else 0.0
    
    # Trace-level classification (if we have access to trace information)
    trace_accuracy = None
    if eval_dataset is not None and hasattr(eval_dataset, 'trace_groups'):
        # Group by trace and classify each trace
        trace_predictions = {}  # trace_id -> (has_mistake, max_energy)
        trace_labels = {}  # trace_id -> (has_mistake, mistake_index)
        
        for trace_id, indices in eval_dataset.trace_groups.items():
            trace_energies_list = []
            trace_has_mistake = False
            trace_mistake_idx = -1
            
            for idx in indices:
                if idx < len(energies):
                    trace_energies_list.append(energies[idx])
                    # Check if this trace has a mistake
                    example = eval_dataset.examples[idx]
                    if example.mistake_index is not None and example.mistake_index >= 0:
                        trace_has_mistake = True
                        trace_mistake_idx = example.mistake_index
            
            if trace_energies_list:
                # avg_energy = np.mean(trace_energies_list)
                max_energy = np.max(trace_energies_list)
                trace_predictions[trace_id] = (max_energy, trace_mistake_idx >= 0)
                trace_labels[trace_id] = trace_has_mistake
        
        # Use tuned threshold if available (from ThresholdTuningCallback), otherwise tune on eval_dataset
        if len(trace_predictions) > 0:
            if tuned_threshold is not None:
                # Use threshold tuned on threshold_tuning_dataset
                threshold = tuned_threshold
            else:
                # Fallback: tune threshold on eval_dataset (not ideal but functional)
                threshold = _tune_threshold_on_traces(trace_predictions, trace_labels)
            
            correct = 0
            total = 0
            for trace_id in trace_predictions:
                pred_has_mistake = trace_predictions[trace_id][0] > threshold
                true_has_mistake = trace_labels[trace_id]
                if pred_has_mistake == true_has_mistake:
                    correct += 1
                total += 1
            
            trace_accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        # Energy separation (primary training objective)
        'energy_gap': energy_gap,
        'mean_energy_positive': mean_energy_pos,
        'mean_energy_negative': mean_energy_neg,
        'pos_separation_rate': pos_separation_rate,  # % of positive examples below median
        
        # Overall statistics
        'mean_energy': float(np.mean(energies)),
        'std_energy': float(np.std(energies)),
        
        # Trace-level classification (if available)
    }
    
    if trace_accuracy is not None:
        metrics['trace_classification_accuracy'] = trace_accuracy
    
    # Keep step-level metrics for backward compatibility, but note they're not the primary objective
    threshold = np.median(energies)
    pred_labels = (energies < threshold).astype(int)
    step_accuracy = accuracy_score(is_positive, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        is_positive, pred_labels, average='binary', zero_division=0
    )
    
    metrics.update({
        'step_accuracy': step_accuracy,  # Note: not the primary training objective
        'step_precision': precision,
        'step_recall': recall,
        'step_f1': f1,
    })
    
    return metrics


class ThresholdTuningCallback(TrainerCallback):
    """Callback to tune threshold on threshold_tuning_dataset before evaluation."""
    
    def __init__(self, threshold_tuning_dataset, threshold_tuning_ratio=0.2):
        self.threshold_tuning_dataset = threshold_tuning_dataset
        self.tuned_threshold = None
        self.threshold_tuning_ratio = threshold_tuning_ratio
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Tune threshold on threshold_tuning_dataset before evaluation."""
        if model is not None and self.threshold_tuning_dataset is not None:
            # Run inference on threshold_tuning_dataset
            model.eval()
            all_energies = []
            trace_energies_dict = {}  # trace_id -> list of energies
            trace_labels_dict = {}  # trace_id -> has_mistake
            
            with torch.no_grad():
                from torch.utils.data import DataLoader
                dataloader = DataLoader(
                    self.threshold_tuning_dataset,
                    batch_size=args.per_device_eval_batch_size,
                    shuffle=False
                )
                
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(model.device)
                    attention_mask = batch['attention_mask'].to(model.device)
                    example_indices = batch['example_idx']
                    
                    energies = model(input_ids=input_ids, attention_mask=attention_mask)
                    energies_np = energies.cpu().numpy()
                    all_energies.extend(energies_np)
                    
                    # Group by trace
                    for i, idx in enumerate(example_indices):
                        example = self.threshold_tuning_dataset.examples[idx.item()]
                        trace_id = example.trace_id
                        
                        if trace_id not in trace_energies_dict:
                            trace_energies_dict[trace_id] = []
                            trace_labels_dict[trace_id] = (
                                example.mistake_index is not None and example.mistake_index >= 0
                            )
                        
                        trace_energies_dict[trace_id].append(energies_np[i])
            
            # Compute max energy per trace
            trace_predictions = {}
            for trace_id, energies_list in trace_energies_dict.items():
                max_energy = np.max(energies_list)
                trace_predictions[trace_id] = (max_energy, trace_labels_dict[trace_id])
            
            # Tune threshold
            if len(trace_predictions) > 0:
                self.tuned_threshold = _tune_threshold_on_traces(trace_predictions, trace_labels_dict)
            else:
                self.tuned_threshold = np.median(all_energies) if len(all_energies) > 0 else 0.0
            
            model.train()


class WandbLoggingCallback(TrainerCallback):
    """Custom callback for additional wandb logging."""
    
    def __init__(self):
        super().__init__()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log additional metrics to wandb."""
        if logs is not None and wandb.run is not None:
            if 'learning_rate' in logs:
                wandb.log({'train/learning_rate': logs['learning_rate']}, step=state.global_step)


def init_wandb(
    held_out_task: str,
    model_name: str,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    warmup_steps: int,
    train_examples_count: int,
    val_examples_count: int,
    delta_1: float,
    delta_2: float,
    delta_3: float,
    delta_4: float,
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
    lambda_4: float,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    resume_run_id: Optional[str] = None
):
    """
    Initialize Weights & Biases logging.
    """
    if wandb_run_name is None:
        wandb_run_name = f"energy_model_heldout_{held_out_task}"
    
    config = {
        "held_out_task": held_out_task,
        "model_name": model_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "train_examples": train_examples_count,
        "val_examples": val_examples_count,
        "training_steps": TRAINING_STEPS.get(held_out_task, max_steps),
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "delta_4": delta_4,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "lambda_3": lambda_3,
        "lambda_4": lambda_4,
    }
    
    if resume_run_id:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=resume_run_id,
            resume="must",
            name=wandb_run_name,
            config=config
        )
    else:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=config
        )
    
    return run


def train_energy_model(
    held_out_task: str,
    model_name: str,
    data_dir: str,
    output_dir: str,
    max_steps: int = MAX_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    warmup_steps: int = WARMUP_STEPS,
    use_table_steps: bool = True,
    delta_1: float = DELTA_1,
    delta_2: float = DELTA_2,
    delta_3: float = DELTA_3,
    delta_4: float = DELTA_4,
    lambda_1: float = LAMBDA_1,
    lambda_2: float = LAMBDA_2,
    lambda_3: float = LAMBDA_3,
    lambda_4: float = LAMBDA_4,
    pos_len_no_mistake_only: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    resume_run_id: Optional[str] = None,
    use_wandb: bool = True
):
    """
    Train an energy-based mistake-finding model for a specific held-out task.
    
    Args:
        held_out_task: Task to hold out for evaluation
        model_name: Name/path of the base model (PaLM 2 Otter)
        data_dir: Directory containing task data files
        output_dir: Directory to save the trained model
        max_steps: Maximum number of training steps
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps for linear ramp
        delta_1, delta_2, delta_3, delta_4: Margin hyperparameters
        lambda_1, lambda_2, lambda_3, lambda_4: Loss weights
    """
    print(f"\n{'='*80}")
    print(f"Training energy model with held-out task: {held_out_task}")
    print(f"{'='*80}\n")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model and wrap with energy head
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        config = base_model.config
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
        print("Loaded model as AutoModelForCausalLM")
    except Exception as e:
        print(f"Could not load as causal LM: {e}")
        print("Trying to load as generic AutoModel...")
        base_model = AutoModel.from_pretrained(model_name)
        config = base_model.config
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else getattr(config, 'n_embd', 768)
        print("Loaded model as AutoModel")
    
    # Create energy model
    model = EnergyModel(base_model, hidden_size)
    print("Created energy model with scalar energy output")
    
    # Create train/val split
    print("Creating train/val split...")
    train_examples, val_examples = create_train_val_split(
        all_tasks=TASKS,
        held_out_task=held_out_task,
        data_dir=data_dir
    )
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")
    
    # Split validation set equally into metrics reporting (50%) and threshold tuning (50%)
    random.seed(42)
    random.shuffle(val_examples)
    val_threshold_size = len(val_examples) // 2  # 50% for threshold tuning
    val_threshold_examples = val_examples[:val_threshold_size]
    val_metrics_examples = val_examples[val_threshold_size:]
    
    print(f"Val metrics examples: {len(val_metrics_examples)}")
    print(f"Val threshold tuning examples: {len(val_threshold_examples)}")
    
    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        wandb_run = init_wandb(
            held_out_task=held_out_task,
            model_name=model_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            train_examples_count=len(train_examples),
            val_examples_count=len(val_examples),
            delta_1=delta_1,
            delta_2=delta_2,
            delta_3=delta_3,
            delta_4=delta_4,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            resume_run_id=resume_run_id
        )
        print(f"Initialized wandb run: {wandb_run.name} (ID: {wandb_run.id})")
    
    # Create datasets
    train_dataset = EnergyModelDataset(train_examples, tokenizer)
    val_metrics_dataset = EnergyModelDataset(val_metrics_examples, tokenizer)
    val_threshold_dataset = EnergyModelDataset(val_threshold_examples, tokenizer)
    
    # Determine max steps
    if use_table_steps:
        actual_max_steps = TRAINING_STEPS.get(held_out_task, max_steps)
        print(f"Training for up to {actual_max_steps} steps (as specified for {held_out_task} in Table 8)")
    else:
        actual_max_steps = max_steps
        print(f"Training for up to {max_steps} steps, will select best checkpoint")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=actual_max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb" if use_wandb else "none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )
    
    # Create threshold tuning callback
    threshold_callback = ThresholdTuningCallback(val_threshold_dataset)
    
    # Create trainer with callbacks
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3), threshold_callback]
    callbacks = [threshold_callback]
    if use_wandb:
        callbacks.append(WandbLoggingCallback())
    
    # Create a closure to pass eval_dataset and access tuned threshold
    def make_compute_metrics(eval_ds, threshold_cb):
        def compute_metrics(eval_pred):
            tuned_thresh = threshold_cb.tuned_threshold if hasattr(threshold_cb, 'tuned_threshold') else None
            return compute_energy_metrics(eval_pred, eval_dataset=eval_ds, tuned_threshold=tuned_thresh)
        return compute_metrics
    
    trainer = EnergyModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_metrics_dataset,
        compute_metrics=make_compute_metrics(val_metrics_dataset, threshold_callback),
        callbacks=callbacks,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=delta_3,
        delta_4=delta_4,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        lambda_4=lambda_4,
        pos_len_no_mistake_only=pos_len_no_mistake_only,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save energy head separately
    torch.save(model.energy_head.state_dict(), os.path.join(output_dir, "energy_head.pt"))
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    # Log final results to wandb
    if use_wandb and wandb_run:
        wandb.log({
            "final/eval_loss": eval_results.get("eval_loss", 0),
            "final/eval_accuracy": eval_results.get("eval_accuracy", 0),
            "final/eval_precision": eval_results.get("eval_precision", 0),
            "final/eval_recall": eval_results.get("eval_recall", 0),
            "final/eval_f1": eval_results.get("eval_f1", 0),
            "final/mean_energy": eval_results.get("eval_mean_energy", 0),
            "final/energy_gap": eval_results.get("eval_energy_gap", 0),
            "final/total_steps": trainer.state.global_step,
        })
        
        if trainer.state.best_model_checkpoint:
            wandb.config.update({
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
            })
    
    # Save evaluation results
    with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Finish wandb run
    if use_wandb and wandb_run:
        wandb.finish()
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train energy-based mistake-finding models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/palm-2-otter",
        help="Name or path of the base model (PaLM 2 Otter)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing task data files (task_name.jsonl format)"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./checkpoints",
        help="Base directory for saving trained models"
    )
    parser.add_argument(
        "--held_out_task",
        type=str,
        choices=TASKS,
        help="Task to hold out for evaluation. If not specified, trains for all tasks."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--train_full_20k",
        action="store_true",
        help="Train for full 20k steps and select best checkpoint (instead of using Table 8 step counts)"
    )
    parser.add_argument(
        "--delta_1",
        type=float,
        default=DELTA_1,
        help="Margin for positive-negative contrast (must be > 0)"
    )
    parser.add_argument(
        "--delta_2",
        type=float,
        default=DELTA_2,
        help="Margin for negative length-based contrast (>= 0)"
    )
    parser.add_argument(
        "--delta_3",
        type=float,
        default=DELTA_3,
        help="Margin for positive length-based contrast (>= 0)"
    )
    parser.add_argument(
        "--delta_4",
        type=float,
        default=DELTA_4,
        help="Margin for energy-jump loss (must be > 0)"
    )
    parser.add_argument(
        "--lambda_1",
        type=float,
        default=LAMBDA_1,
        help="Weight for positive-negative contrast loss"
    )
    parser.add_argument(
        "--lambda_2",
        type=float,
        default=LAMBDA_2,
        help="Weight for negative length-based contrast loss"
    )
    parser.add_argument(
        "--lambda_3",
        type=float,
        default=LAMBDA_3,
        help="Weight for positive length-based contrast loss"
    )
    parser.add_argument(
        "--lambda_4",
        type=float,
        default=LAMBDA_4,
        help="Weight for energy-jump loss"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="energy-model-mistake-finding",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Custom name for wandb run (auto-generated if not provided)"
    )
    parser.add_argument(
        "--resume_run_id",
        type=str,
        default=None,
        help="W&B run ID to resume (for resuming interrupted training)"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--pos_len_no_mistake_only",
        action="store_true",
        help="Only calculate positive length-based contrast (loss 3) for examples with no mistake"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Determine which tasks to train
    if args.held_out_task:
        tasks_to_train = [args.held_out_task]
    else:
        tasks_to_train = TASKS
    
    # Train an energy model for each held-out task
    for held_out_task in tasks_to_train:
        output_dir = os.path.join(
            args.output_base_dir,
            f"energy_model_heldout_{held_out_task}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Generate run name if not provided
            run_name = args.wandb_run_name
            if run_name is None and not args.no_wandb:
                run_name = f"energy_model_heldout_{held_out_task}"
            
            train_energy_model(
                held_out_task=held_out_task,
                model_name=args.model_name,
                data_dir=args.data_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_table_steps=not args.train_full_20k,
                delta_1=args.delta_1,
                delta_2=args.delta_2,
                delta_3=args.delta_3,
                delta_4=args.delta_4,
                lambda_1=args.lambda_1,
                lambda_2=args.lambda_2,
                lambda_3=args.lambda_3,
                lambda_4=args.lambda_4,
                pos_len_no_mistake_only=args.pos_len_no_mistake_only,
                wandb_project=args.wandb_project if not args.no_wandb else None,
                wandb_entity=args.wandb_entity if not args.no_wandb else None,
                wandb_run_name=run_name if not args.no_wandb else None,
                resume_run_id=args.resume_run_id if not args.no_wandb else None,
                use_wandb=not args.no_wandb
            )
            print(f"\n✓ Successfully trained energy model for held-out task: {held_out_task}")
        except Exception as e:
            print(f"\n✗ Error training energy model for {held_out_task}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()

