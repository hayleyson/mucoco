"""
Training script for mistake-finding classifiers as described in the paper.

Fine-tunes PaLM 2 Otter on 4 out of 5 tasks, holding out one task for evaluation.
This is done for each of the 5 tasks.

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
import random
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
# from datasets import load_dataset, Dataset as HFDataset  # Optional: for loading from HuggingFace datasets

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

@dataclass
class CoTStepExample:
    """Represents a single CoT step for training."""
    task: str
    input_text: str
    previous_steps: str  # All steps before the current one
    current_step: str  # The current step being evaluated
    label: int  # 1 if correct, 0 if incorrect
    step_index: int  # Index of this step in the CoT trace


def process_cot_steps(
    task: str,
    input_text: str,
    steps: List[str],
    mistake_index: Optional[int],
    include_post_mistake: bool = True
) -> List[CoTStepExample]:
    """
    Process a list of CoT steps into training examples.
    
    Args:
        task: Name of the task
        input_text: The input/problem text
        steps: List of step strings (already parsed)
        mistake_index: Index of the first step where a mistake occurred (None if no mistake)
        include_post_mistake: If True, include steps after first mistake as incorrect
    
    Returns:
        List of CoTStepExample objects
    """
    if not steps:
        return []
    
    examples = []
    
    # Create examples for each step
    previous_steps_text = ""
    for i, step_text in enumerate(steps):
        # Determine the label for this step
        # Steps before mistake_index are correct (label=1)
        # Steps at and after mistake_index are incorrect (label=0)
        if mistake_index is None:
            # No mistake, all steps are correct
            label = 1
        elif i < mistake_index:
            # Step is before the first mistake, so it's correct
            label = 1
        else:
            # Step is at or after the first mistake, so it's incorrect
            label = 0
        
        # Create the example
        example = CoTStepExample(
            task=task,
            input_text=input_text,
            previous_steps=previous_steps_text,
            current_step=step_text,
            label=label,
            step_index=i
        )
        examples.append(example)
        
        # Update previous_steps_text for next iteration
        if previous_steps_text:
            previous_steps_text += f"\nStep {i+1}: {step_text}"
        else:
            previous_steps_text = f"Step {i+1}: {step_text}"
    
    return examples


def format_input_for_classifier(example: CoTStepExample) -> str:
    """
    Format an example into the input text for the classifier.
    
    The classifier should predict whether a CoT step is correct, given:
    - The task
    - The input/problem
    - Previous steps
    - The current step
    """
    # Format: Task: <task>\nInput: <input>\n<previous_steps>\nStep: <current_step>
    parts = [f"Task: {example.task}"]
    parts.append(f"Input: {example.input_text}")
    
    if example.previous_steps:
        parts.append(example.previous_steps)
    
    parts.append(f"Step: {example.current_step}")
    
    return "\n".join(parts)


class MistakeFindingDataset(Dataset):
    """Dataset for mistake-finding classification."""
    
    def __init__(
        self,
        examples: List[CoTStepExample],
        tokenizer: AutoTokenizer,
        max_length: Optional[int] = None
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        # Use model's max length if not specified, but cap it to a reasonable value for training
        if max_length is None:
            # Get max length from tokenizer
            self.max_length = getattr(tokenizer, 'model_max_length', None)
            if self.max_length is None:
                # Fallback to 2048: a common sequence length for many transformer models
                self.max_length = 2048
                print(f"Warning: Tokenizer does not have model_max_length. Using default: {self.max_length}")
            elif self.max_length > 8192:
                # Cap very large max_lengths (e.g., 128k for Llama 3.1) to prevent OOM
                # Models may support very long sequences for inference, but training requires much more memory
                # 8192 is a reasonable upper limit that balances memory and sequence length
                original_max = self.max_length
                self.max_length = 8192
                print(f"Warning: Tokenizer model_max_length ({original_max}) is very large. Capping to {self.max_length} for training to prevent OOM.")
        else:
            self.max_length = max_length
            print(f"Using specified max_length: {self.max_length}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = format_input_for_classifier(example)
        
        # Don't pad here - padding will be done dynamically per batch
        # Only truncate if the sequence is longer than max_length
        encoding = self.tokenizer(
            text,
            truncation=True if self.max_length else False,
            max_length=self.max_length if self.max_length else None,
            return_tensors=None  # Return as lists, not tensors
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': example.label
        }


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


# def create_train_val_split(
#     all_tasks: List[str],
#     held_out_task: str,
#     data_dir: str
# ) -> Tuple[List[CoTStepExample], List[CoTStepExample]]:
#     """
#     Create train/val split by holding out one task.
    
#     Args:
#         all_tasks: List of all task names
#         held_out_task: Task to hold out for validation
#         data_dir: Directory containing task data files
    
#     Returns:
#         Tuple of (train_examples, val_examples)
#     """
#     train_examples = []
#     val_examples = []
    
#     for task in all_tasks:
#         # Load data for this task
#         data_path = os.path.join(data_dir, f"{task}.jsonl")
#         if not os.path.exists(data_path):
#             print(f"Warning: Data file not found for task {task}: {data_path}")
#             continue
        
#         task_data = load_task_data(task, data_path)
        
#         for example_data in task_data:
#             input_text = example_data.get('input', '')
#             steps = example_data.get('steps', [])
#             mistake_index = example_data.get('mistake_index', None)
            
#             # Handle None mistake_index (no mistake, all steps correct)
#             if mistake_index is not None and not isinstance(mistake_index, int):
#                 # Skip invalid examples
#                 continue
            
#             if not steps:
#                 continue
            
#             # Process steps into step examples
#             step_examples = process_cot_steps(
#                 task=task,
#                 input_text=input_text,
#                 steps=steps,
#                 mistake_index=mistake_index,
#                 include_post_mistake=True
#             )
            
#             if task == held_out_task:
#                 val_examples.extend(step_examples)
#             else:
#                 train_examples.extend(step_examples)
    
#     return train_examples, val_examples

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


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class WandbLoggingCallback(TrainerCallback):
    """Custom callback for additional wandb logging."""
    
    def __init__(self):
        super().__init__()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log additional metrics to wandb."""
        if logs is not None and wandb.run is not None:
            # The Trainer already logs standard metrics, but we can add custom ones here
            # For example, log learning rate at each step
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
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    resume_run_id: Optional[str] = None
):
    """
    Initialize Weights & Biases logging.
    
    Args:
        held_out_task: Task held out for evaluation
        model_name: Name of the base model
        batch_size: Training batch size
        learning_rate: Learning rate
        max_steps: Maximum training steps
        warmup_steps: Number of warmup steps
        train_examples_count: Number of training examples
        val_examples_count: Number of validation examples
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        wandb_run_name: Custom run name (if None, auto-generated)
        resume_run_id: W&B run ID to resume (if resuming a run)
    
    Returns:
        wandb run object
    """
    if wandb_run_name is None:
        wandb_run_name = f"mistake_classifier_heldout_{held_out_task}"
    
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


def train_classifier(
    held_out_task: str,
    model_name: str,
    data_dir: str,
    output_dir: str,
    max_steps: int = MAX_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    warmup_steps: int = WARMUP_STEPS,
    max_length: Optional[int] = None,
    use_table_steps: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    resume_run_id: Optional[str] = None,
    use_wandb: bool = True,
    gradient_accumulation_steps: Optional[int] = None,
    use_memory_efficient: bool = True,
):
    """
    Train a mistake-finding classifier for a specific held-out task.
    
    Args:
        held_out_task: Task to hold out for evaluation
        model_name: Name/path of the base model (PaLM 2 Otter)
        data_dir: Directory containing task data files
        output_dir: Directory to save the trained model
        max_steps: Maximum number of training steps
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps for linear ramp
    """
    print(f"\n{'='*80}")
    print(f"Training classifier with held-out task: {held_out_task}")
    print(f"{'='*80}\n")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set pad_token_id in tokenizer config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Memory-efficient loading options
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",  # Automatically distribute model across available devices
    }
    
    # Try to load as sequence classification model first
    # If that fails (e.g., for causal LMs like Llama), load base model and add classification head
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            **model_kwargs
        )
        # Set pad_token_id in model config
        model.config.pad_token_id = tokenizer.pad_token_id
        print("Loaded model as AutoModelForSequenceClassification")
    except (ValueError, OSError, KeyError) as e:
        print(f"Could not load as sequence classification model: {e}")
        print("Attempting to load base model and add classification head...")
        
        # Load the base model (could be causal LM or encoder)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            # Set pad_token_id in model config
            base_model.config.pad_token_id = tokenizer.pad_token_id
            config = base_model.config
            # Get the hidden size from the model
            if hasattr(base_model, 'config'):
                hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
            else:
                hidden_size = 768  # default fallback
            
            # Create a wrapper model with classification head
            class CausalLMForSequenceClassification(nn.Module):
                def __init__(self, base_model, num_labels, hidden_size):
                    super().__init__()
                    self.base_model = base_model
                    self.num_labels = num_labels
                    # Add classification head with same dtype as base model
                    # Get dtype from base model
                    model_dtype = next(base_model.parameters()).dtype
                    self.classifier = nn.Linear(hidden_size, num_labels, dtype=model_dtype)
                    # Enable gradient checkpointing if available
                    if hasattr(base_model, 'gradient_checkpointing_enable'):
                        base_model.gradient_checkpointing_enable()
                    # Freeze base model if desired (uncomment to freeze)
                    # for param in self.base_model.parameters():
                    #     param.requires_grad = False
                    
                def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                    # Get outputs from the base model
                    # Set output_hidden_states=True to get hidden states
                    outputs = self.base_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    
                    # Get the last hidden state
                    # For causal LMs, hidden_states is a tuple where last element is the final layer
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        last_hidden_state = outputs.hidden_states[-1]
                    else:
                        # Fallback: try to get from base_model output
                        last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                    
                    # For causal LMs, we use the last non-padding token's hidden state
                    if attention_mask is not None:
                        # Find the last non-padding token for each sequence
                        sequence_lengths = attention_mask.sum(dim=1) - 1
                        # Ensure sequence_lengths are within bounds
                        sequence_lengths = torch.clamp(sequence_lengths, 0, last_hidden_state.size(1) - 1)
                        pooled_output = last_hidden_state[range(len(sequence_lengths)), sequence_lengths]
                    else:
                        # Use the last token
                        pooled_output = last_hidden_state[:, -1, :]
                    
                    logits = self.classifier(pooled_output)
                    
                    loss = None
                    if labels is not None:
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    
                    from transformers.modeling_outputs import SequenceClassifierOutput
                    return SequenceClassifierOutput(
                        loss=loss,
                        logits=logits,
                        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                    )
                
                def save_pretrained(self, save_directory):
                    import os
                    os.makedirs(save_directory, exist_ok=True)
                    self.base_model.save_pretrained(save_directory)
                    # Save classifier weights separately
                    torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
                    # Save config
                    config = self.base_model.config
                    config.num_labels = self.num_labels
                    config.save_pretrained(save_directory)
            
            model = CausalLMForSequenceClassification(base_model, num_labels=2, hidden_size=hidden_size)
            print("Created classification model from causal LM base model")
            
        except Exception as e2:
            print(f"Failed to load as causal LM: {e2}")
            print("Trying to load as generic AutoModel...")
            # Last resort: try AutoModel
            base_model = AutoModel.from_pretrained(
                model_name,
                **model_kwargs
            )
            # Set pad_token_id in model config
            base_model.config.pad_token_id = tokenizer.pad_token_id
            config = base_model.config
            hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else getattr(config, 'n_embd', 768)
            
            class GenericModelForSequenceClassification(nn.Module):
                def __init__(self, base_model, num_labels, hidden_size):
                    super().__init__()
                    self.base_model = base_model
                    self.num_labels = num_labels
                    # Get dtype from base model
                    model_dtype = next(base_model.parameters()).dtype
                    self.classifier = nn.Linear(hidden_size, num_labels, dtype=model_dtype)
                    # Enable gradient checkpointing if available
                    if hasattr(base_model, 'gradient_checkpointing_enable'):
                        base_model.gradient_checkpointing_enable()
                    
                def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                    
                    # Use pooled output or last hidden state
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        pooled_output = outputs.pooler_output
                    else:
                        last_hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                        if attention_mask is not None:
                            sequence_lengths = attention_mask.sum(dim=1) - 1
                            pooled_output = last_hidden_state[range(len(sequence_lengths)), sequence_lengths]
                        else:
                            pooled_output = last_hidden_state[:, -1, :]
                    
                    logits = self.classifier(pooled_output)
                    
                    loss = None
                    if labels is not None:
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    
                    from transformers.modeling_outputs import SequenceClassifierOutput
                    return SequenceClassifierOutput(
                        loss=loss,
                        logits=logits,
                        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                    )
                
                def save_pretrained(self, save_directory):
                    self.base_model.save_pretrained(save_directory)
                    import os
                    torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
            
            model = GenericModelForSequenceClassification(base_model, num_labels=2, hidden_size=hidden_size)
            print("Created classification model from generic base model")
    
    
    
    # Create train/val split
    print("Creating train/val split...")
    train_examples, val_examples = create_train_val_split(
        all_tasks=TASKS,
        held_out_task=held_out_task,
        data_dir=data_dir
    )
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")
    
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
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            resume_run_id=resume_run_id
        )
        print(f"Initialized wandb run: {wandb_run.name} (ID: {wandb_run.id})")
    
    # For dynamic padding, we still need a max_length for truncation to prevent OOM
    # But we'll use it only for truncation, not for padding
    # Compute actual max sequence length in the dataset to set truncation limit
    print("Computing actual max sequence length in dataset for truncation limit...")
    max_seq_len = 0
    
    # Check all examples to find the true maximum (important to not truncate)
    # For very large datasets, we could sample, but for safety we check all
    print(f"Checking {len(train_examples)} training examples...")
    for i, example in enumerate(train_examples):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(train_examples)} examples, current max: {max_seq_len}")
        text = format_input_for_classifier(example)
        # Use encode to get actual token count
        tokens = tokenizer.encode(text, add_special_tokens=True)
        seq_len = len(tokens)
        max_seq_len = max(max_seq_len, seq_len)
        if seq_len == 32425:
            print(text)
    
    # Also check validation examples
    print(f"Checking {len(val_examples)} validation examples...")
    for i, example in enumerate(val_examples):
        text = format_input_for_classifier(example)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        seq_len = len(tokens)
        max_seq_len = max(max_seq_len, seq_len)
        if seq_len == 32425:
            print(text)
    
    # Add some padding (10% or 128 tokens, whichever is larger) to account for variations
    padding = max(int(max_seq_len * 0.1), 128)
    truncation_max_length = max_seq_len + padding
    
    # Round up to nearest 64 for efficiency (many models work better with multiples of 64)
    truncation_max_length = ((truncation_max_length + 63) // 64) * 64
    if max_length is not None:
        truncation_max_length = min(truncation_max_length, max_length)
    
    print(f"\nDataset max sequence length: {max_seq_len} tokens")
    print(f"Specified max_length argument: {max_length} tokens")
    print(f"Using truncation max_length: {truncation_max_length} tokens (for truncation only, padding will be dynamic per batch)")
    
    # Create datasets with truncation max_length (but no fixed padding)
    train_dataset = MistakeFindingDataset(train_examples, tokenizer, max_length=truncation_max_length)
    val_dataset = MistakeFindingDataset(val_examples, tokenizer, max_length=truncation_max_length)
    
    # Create data collator for dynamic padding
    # This will pad each batch to the longest sequence in that batch
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,  # Dynamic padding to longest in batch
        return_tensors="pt"
    )
    
    # According to the paper: "All 5 models are fine-tuned for 20k steps"
    # But Table 8 shows the actual number of steps trained for each model
    # If use_table_steps=True, we use the step count from Table 8
    # If use_table_steps=False, we train for full 20k and select best checkpoint
    if use_table_steps:
        actual_max_steps = TRAINING_STEPS.get(held_out_task, max_steps)
        print(f"Training for up to {actual_max_steps} steps (as specified for {held_out_task} in Table 8)")
    else:
        actual_max_steps = max_steps
        print(f"Training for up to {max_steps} steps, will select best checkpoint")
    
    # Training arguments
    # Note: lr_scheduler_type="cosine" with warmup_steps implements:
    # - Linear ramp (warmup) for the first warmup_steps
    # - Cosine decay for the remaining steps
    
    # Adjust batch size and gradient accumulation for large models
    # If using device_map="auto", we need to handle batch size differently
    effective_batch_size = batch_size
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 1
    
    # For large models, reduce per-device batch size and use gradient accumulation
    if gradient_accumulation_steps == 1:
        # Reduce batch size for large models
        effective_batch_size = min(batch_size, MINIMUM_BATCH_SIZE)  # Cap at 4 for large models
        # Use gradient accumulation to maintain effective batch size
        gradient_accumulation_steps = max(1, batch_size // effective_batch_size)
        print(f"Using memory-efficient settings: batch_size={effective_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
    elif gradient_accumulation_steps is not None and gradient_accumulation_steps > 1:
        print(f"Using specified gradient_accumulation_steps={gradient_accumulation_steps}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # We use max_steps instead
        max_steps=actual_max_steps,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",  # Cosine decay after linear warmup
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
        # Use bf16 if supported, otherwise fp16
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        # Enable gradient checkpointing to save memory
        gradient_checkpointing=True,
        # Optimize memory usage
        dataloader_pin_memory=False,  # Can cause OOM on some systems
        remove_unused_columns=False,  # Keep all columns for compatibility
    )
    
    # Create trainer with callbacks
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    callbacks = []
    if use_wandb:
        callbacks.append(WandbLoggingCallback())
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        data_collator=data_collator  # Use dynamic padding collator
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    # Log final results to wandb
    if use_wandb and wandb_run:
        # Log final metrics
        wandb.log({
            "final/eval_loss": eval_results.get("eval_loss", 0),
            "final/eval_accuracy": eval_results.get("eval_accuracy", 0),
            "final/eval_precision": eval_results.get("eval_precision", 0),
            "final/eval_recall": eval_results.get("eval_recall", 0),
            "final/eval_f1": eval_results.get("eval_f1", 0),
            "final/total_steps": trainer.state.global_step,
        })
        
        # Log best checkpoint info
        if trainer.state.best_model_checkpoint:
            wandb.config.update({
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
            })
        
        # Optionally log model directory as artifact
        # Note: This requires the model to be saved first
        # artifact = wandb.Artifact(
        #     name=f"model_{held_out_task}",
        #     type="model",
        #     description=f"Mistake-finding classifier for held-out task: {held_out_task}"
        # )
        # artifact.add_dir(output_dir)
        # wandb.log_artifact(artifact, aliases=["latest"])
    
    # Save evaluation results
    with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Finish wandb run
    if use_wandb and wandb_run:
        wandb.finish()
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train mistake-finding classifiers")
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
        "--wandb_project",
        type=str,
        default="mistake-finding-classifier",
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
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps (auto-calculated for large models if not specified)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length for truncation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Determine which tasks to train
    if args.held_out_task:
        tasks_to_train = [args.held_out_task]
    else:
        tasks_to_train = TASKS
    
    # Train a classifier for each held-out task
    for held_out_task in tasks_to_train:
        output_dir = os.path.join(
            args.output_base_dir,
            f"classifier_heldout_{held_out_task}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Generate run name if not provided
            run_name = args.wandb_run_name
            if run_name is None and not args.no_wandb:
                run_name = f"mistake_classifier_heldout_{held_out_task}"
            
            train_classifier(
                held_out_task=held_out_task,
                model_name=args.model_name,
                data_dir=args.data_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                use_table_steps=not args.train_full_20k,
                wandb_project=args.wandb_project if not args.no_wandb else None,
                wandb_entity=args.wandb_entity if not args.no_wandb else None,
                wandb_run_name=run_name if not args.no_wandb else None,
                resume_run_id=args.resume_run_id if not args.no_wandb else None,
                use_wandb=not args.no_wandb,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
            print(f"\n✓ Successfully trained classifier for held-out task: {held_out_task}")
        except Exception as e:
            print(f"\n✗ Error training classifier for {held_out_task}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()

