"""
Evaluate LLM span detection results by comparing predictions with gold labels.

This script:
1. Loads gold labels from original data files
2. Loads predictions from post-processed LLM results
3. Calculates metrics (precision, recall, F1, AP, RR) using functions from evaluate_locate_nli.py
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import re

from new_module.evaluation.locate_metrics_utils import (
    apply_ap,
    apply_precision,
    apply_recall,
    apply_f1,
    get_rr,
    calculate_exact_match,
    binarize_labels,
    convert_gpt2_token_labels_to_word_labels,
)



def parse_filename_metadata(filename: str) -> Tuple[str, str]:
    """
    Parse filename to extract model name and prompt type.
    
    Args:
        filename: The filename (e.g., "gpt-4.1-2025-04-14_locate_toxic_0shot_type1_v3_toxicspans_1762982031_processed.jsonl")
    
    Returns:
        Tuple of (model, prompt_type)
    """
    # Extract model name (everything before _locate_)
    if '_locate_' in filename:
        model = filename.split('_locate_')[0]
    else:
        model = 'unknown'
    
    # Extract prompt type (everything before _toxicspans_ or _inconsistentspans_, including shot count)
    prompt_type = 'default'
    
    # Look for pattern before toxicspans or inconsistentspans
    # Pattern: ..._something_toxicspans_... or ..._something_inconsistentspans_...
    # This captures everything from the shot pattern to just before the dataset name
    # Examples: 0shot_type1_v3, 0shot_v2, 5shot_cot_type1_v4
    match_toxic = re.search(r'_(\d+shot.*?)_toxicspans_', filename)
    match_incon = re.search(r'_(\d+shot.*?)_inconsistentspans_', filename)
    
    if match_toxic:
        prompt_type = match_toxic.group(1)
    elif match_incon:
        prompt_type = match_incon.group(1)
    
    return model, prompt_type




def load_gold_labels_toxic(original_file: str) -> List[Dict]:
    """
    Load gold labels for toxic span detection.
    Converts token-level labels to word-level labels.
    """
     
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prefix = data.get('prefix', '')
            generation = data['generation']
            words = data['generation_words']
            word_labels_continuous = data['generation_word_labels']
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(word_labels_continuous, threshold=0.5)
            
            gold_data.append({
                'index': len(gold_data),
                'prefix': prefix,
                'generation': generation,
                'generation_words': words,
                'word_labels_continuous': word_labels_continuous,
                'word_labels_binary': word_labels_binary
            })
    
    return gold_data




def load_gold_labels_toxic_extended(original_file: str) -> List[Dict]:
    """
    Load gold labels for toxic span detection.
    Converts token-level labels to word-level labels.
    """
    
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prefix = data.get('prefix', '')
            generation = data['text']
            words = data['words']
            word_labels_continuous = data['word_labels']
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(word_labels_continuous, threshold=0.5)
            
            gold_data.append({
                'index': len(gold_data),
                'prefix': prefix,   
                'generation': generation,
                'generation_words': words,
                'word_labels_continuous': word_labels_continuous,
                'word_labels_binary': word_labels_binary
            })
    
    return gold_data



def load_gold_labels_inconsistent(original_file: str) -> List[Dict]:
    """
    Load gold labels for inconsistent span detection.
    Uses hypothesis_word_labels and binarizes at 0.5.
    """
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pairID = data.get('pairID', '')
            premise = data.get('premise', '')
            hypothesis = data.get('hypothesis', '')
            hypothesis_words = data.get('hypothesis_words', [])
            hypothesis_word_labels = data.get('hypothesis_word_labels', [])
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(hypothesis_word_labels, threshold=0.5)
            
            gold_data.append({
                'index': len(gold_data),
                'pairID': pairID,
                'premise': premise,
                'hypothesis': hypothesis,
                'hypothesis_words': hypothesis_words,
                'word_labels_continuous': hypothesis_word_labels,
                'word_labels_binary': word_labels_binary
            })
    
    return gold_data


def load_gold_labels_bbm(original_file: str) -> List[Dict]:
    """
    Load gold labels for BIG-Bench-Mistake tasks.
    Extracts 'input' and 'mistake_index'.
    """
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            input_text = data.get('input', '')
            mistake_index = data.get('mistake_index')
            
            # Normalize mistake_index
            # In gold: null (no mistake), integer (step of first mistake)
            if mistake_index is None:
                gold_label = "No"
            else:
                gold_label = str(mistake_index)
            
            gold_data.append({
                'index': len(gold_data),
                'input': input_text,
                'gold_label': gold_label
            })
    
    return gold_data


def load_predictions(prediction_file: str) -> List[Dict]:
    """Load predictions from post-processed results file."""
    predictions = []
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_predictions_bbm(prediction_file: str) -> List[Dict]:
    """Load predictions for BBM tasks."""
    predictions = []
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Result might be a single string or a JSON object with 'answer'
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            
            answer = data.get('answer', '')
            reasoning = data.get('reasoning', '')
            
            # Normalize answer
            if isinstance(answer, (int, float)):
                answer = str(int(answer))
            elif isinstance(answer, str):
                answer = answer.strip()
                if answer.lower() in ['no', 'none', 'no mistake', 'n/a', 'nothing'] or answer == '':
                    answer = "No"
            else:
                answer = "No" # Default/Fallback
                
            predictions.append({
                'answer': answer,
                'reasoning': reasoning
            })
    return predictions

def evaluate_predictions(
    gold_data: List[Dict],
    predictions: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate predictions against gold labels.
    
    Returns:
        Dictionary with metrics: mean_precision, mean_recall, mean_f1, mean_ap, mean_rr
    """
    # Create DataFrame for easier computation
    df_data = []
    
    for gold, pred in zip(gold_data, predictions):
        # Get binary labels
        gold_binary = gold['word_labels_binary']
        pred_binary = pred['word_labels']
        
        # For AP and RR, we need scores. Since we only have binary predictions,
        # we'll use the binary predictions as scores (1.0 for positive, 0.0 for negative)
        pred_scores = [float(x) for x in pred_binary]
        
        # Ensure same length
        min_len = min(len(gold_binary), len(pred_binary))
        gold_binary = gold_binary[:min_len]
        pred_binary = pred_binary[:min_len]
        pred_scores = pred_scores[:min_len]
        
        df_data.append({
            'gold_binary': gold_binary,
            'pred_binary': pred_binary,
            'pred_scores': pred_scores
        })
    
    df = pd.DataFrame(df_data)
    
    # Calculate metrics for each example
    df['precision'] = df.apply(
        lambda x: apply_precision(x, 'gold_binary', 'pred_binary'),
        axis=1
    )
    df['recall'] = df.apply(
        lambda x: apply_recall(x, 'gold_binary', 'pred_binary'),
        axis=1
    )
    df['f1'] = df.apply(
        lambda x: apply_f1(x, 'gold_binary', 'pred_binary'),
        axis=1
    )
    df['ap'] = df.apply(
        lambda x: apply_ap(x, 'gold_binary', 'pred_scores'),
        axis=1
    )
    df['rr'] = df.apply(
        lambda x: get_rr(x, 'gold_binary', 'pred_scores'),
        axis=1
    )
    df['exact_match'] = df.apply(
        lambda x: calculate_exact_match(x['gold_binary'], x['pred_binary']),
        axis=1
    )
    
    # Calculate mean metrics
    metrics = {
        'mean_precision': df['precision'].mean(),
        'mean_recall': df['recall'].mean(),
        'mean_f1': df['f1'].mean(),
        'mean_ap': df['ap'].mean(),
        'mean_rr': df['rr'].mean(),
        'exact_match': df['exact_match'].mean(),  # This is already a mean (proportion of exact matches)
        'num_examples': len(df)
    }
    
    return metrics


def evaluate_predictions_bbm(
    gold_data: List[Dict],
    predictions: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate BBM predictions (classification accuracy).
    """
    correct = 0
    correct_no = 0
    correct_mistake = 0
    false_positive_no = 0 # Gold is mistake, Pred is No (False Negative Mistake)
    false_negative_no = 0 # Gold is No, Pred is mistake (False Positive Mistake)
    
    total = len(gold_data)
    
    for gold, pred in zip(gold_data, predictions):
        g = gold['gold_label']
        p = pred['answer']
        
        is_correct = (g == p)
        if is_correct:
            correct += 1
            if g == "No":
                correct_no += 1
            else:
                correct_mistake += 1
        else:
            if g == "No":
                false_negative_no += 1
            else:
                if p == "No":
                    false_positive_no += 1
    
    metrics = {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'correct_no': correct_no,
        'correct_mistake': correct_mistake,
        'false_positive_no': false_positive_no,
        'false_negative_no': false_negative_no,
        'num_examples': total
    }
    
    return metrics


def evaluate_single_file(
    prediction_file: str,
    original_file: str,
    task: str,
    output_file: str = None
):
    """
    Evaluate a single prediction file against gold labels.
    
    Args:
        prediction_file: Path to post-processed prediction file
        original_file: Path to original data file with gold labels
        task: 'toxic' or 'inconsistent'
        output_file: Optional path to save detailed results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {prediction_file}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")
    
    # Load gold labels
    print("Loading gold labels...")
    if task == 'toxic_extended':
        gold_data = load_gold_labels_toxic_extended(original_file)
    elif task == 'toxic':
        gold_data = load_gold_labels_toxic(original_file)
    elif task == 'inconsistent':
        gold_data = load_gold_labels_inconsistent(original_file)
    elif task in ['logical_deduction', 'tracking_shuffled_objects']:
        gold_data = load_gold_labels_bbm(original_file)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    print(f"Loaded {len(gold_data)} gold examples")
    
    # Load predictions
    print("Loading predictions...")
    if task in ['logical_deduction', 'tracking_shuffled_objects']:
        predictions = load_predictions_bbm(prediction_file)
    else:
        predictions = load_predictions(prediction_file)
    print(f"Loaded {len(predictions)} predictions")
    
    # Check if lengths match
    if len(gold_data) != len(predictions):
        warnings.warn(
            f"Length mismatch: gold_data has {len(gold_data)} examples, "
            f"predictions has {len(predictions)} examples. "
            f"Using min({len(gold_data)}, {len(predictions)}) examples."
        )
        min_len = min(len(gold_data), len(predictions))
        gold_data = gold_data[:min_len]
        predictions = predictions[:min_len]
    
    # Evaluate
    print("Calculating metrics...")
    if task in ['logical_deduction', 'tracking_shuffled_objects']:
        metrics = evaluate_predictions_bbm(gold_data, predictions)
    else:
        metrics = evaluate_predictions(gold_data, predictions)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of examples: {metrics['num_examples']}")
    
    if task in ['logical_deduction', 'tracking_shuffled_objects']:
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Correct:         {metrics['correct']} / {metrics['num_examples']}")
        print(f"Correct No:      {metrics['correct_no']}")
        print(f"Correct Mistake: {metrics['correct_mistake']}")
        print(f"Gold No, Pred != No (False Pos Mistake): {metrics['false_negative_no']}")
        print(f"Gold != No, Pred No (False Neg Mistake): {metrics['false_positive_no']}")
    else:
        print(f"Mean Precision:  {metrics['mean_precision']:.4f}")
        print(f"Mean Recall:     {metrics['mean_recall']:.4f}")
        print(f"Mean F1:         {metrics['mean_f1']:.4f}")
        print(f"Mean AP:         {metrics['mean_ap']:.4f}")
        print(f"Mean RR:         {metrics['mean_rr']:.4f}")
        print(f"Exact Match:     {metrics['exact_match']:.4f}")
    print("="*60 + "\n")
    
    # Save detailed results if requested
    if output_file:
        results = {
            'prediction_file': prediction_file,
            'original_file': original_file,
            'task': task,
            'metrics': metrics
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {output_file}\n")
    
    return metrics


def evaluate_all_results(
    results_dir: str,
    original_toxic_extended_file: str,
    original_toxic_file: str,
    original_inconsistent_file: str,
    output_dir: str = None
):
    """
    Evaluate all prediction files in the results directory.
    """
    results_path = Path(results_dir)
    prediction_files = list(results_path.glob('*_processed.jsonl'))
    
    all_metrics = []
    
    for pred_file in prediction_files:
        # Determine task type from filename
        if 'toxicspans_extended' in pred_file.name:
            task = 'toxic_extended'
            original_file = original_toxic_extended_file
        elif 'toxic' in pred_file.name:
            task = 'toxic'
            original_file = original_toxic_file
        elif 'incon' in pred_file.name or 'inconsistent' in pred_file.name:
            task = 'inconsistent'
            original_file = original_inconsistent_file
        elif 'logical_deduction' in pred_file.name:
            task = 'logical_deduction'
            original_file = 'new_module/data/BIG-Bench-Mistake/logical_deduction.jsonl'
        elif 'tracking_shuffled_objects' in pred_file.name:
            task = 'tracking_shuffled_objects'
            original_file = 'new_module/data/BIG-Bench-Mistake/tracking_shuffled_objects.jsonl'
        else:
            print(f"Warning: Could not determine task type for {pred_file.name}, skipping...")
            continue
        
        if original_file is None:
            print(f"Warning: Could not find original file for {pred_file.name}, skipping...")
            continue
        
        # Extract model and prompt_type from filename
        model, prompt_type = parse_filename_metadata(pred_file.name)
        
        # Try to read execution time from corresponding .time file
        execution_seconds = None
        time_file = pred_file.parent / f"{pred_file.stem.replace('_processed', '')}.time"
        if time_file.exists():
            try:
                with open(time_file, 'r') as f:
                    time_content = f.read().strip()
                    # Try to parse as float
                    execution_seconds = float(time_content)
            except (ValueError, IOError) as e:
                print(f"Warning: Could not read execution time from {time_file}: {e}")
        else:
            print(f"Warning: No execution time file found for {pred_file.name}, skipping...")
        
        
        # Evaluate
        try:
            metrics = evaluate_single_file(
                str(pred_file),
                original_file,
                task
            )
            
            metrics['prediction_file'] = str(pred_file)
            metrics['task'] = task
            metrics['model'] = model
            metrics['prompt_type'] = prompt_type
            metrics['execution_seconds'] = execution_seconds
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {pred_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_metrics:
        print("\n" + "="*60)
        print("SUMMARY OF ALL RESULTS")
        print("="*60)
        df_summary = pd.DataFrame(all_metrics)
        display_cols = ['prediction_file', 'model', 'prompt_type', 'task', 'mean_precision', 'mean_recall', 'mean_f1', 'mean_ap', 'mean_rr', 'exact_match', 'accuracy', 'execution_seconds']
        display_cols = [c for c in display_cols if c in df_summary.columns]
        print(df_summary[display_cols].to_string(index=False))
        print("="*60 + "\n")
        
        # Save summary if output_dir is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            summary_file = output_path / 'evaluation_summary.csv'
            df_summary.to_csv(summary_file, index=False)
            print(f"Summary saved to {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM span detection results')
    parser.add_argument('--prediction_file', type=str, default=None,
                        help='Path to post-processed prediction file (single file mode)')
    parser.add_argument('--original_file', type=str, default=None,
                        help='Path to original data file with gold labels (single file mode)')
    parser.add_argument('--task', type=str, choices=['toxic', 'inconsistent', 'logical_deduction', 'tracking_shuffled_objects'], default=None,
                        help='Task type: toxic or inconsistent or logical_deduction or tracking_shuffled_objects (single file mode)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save detailed results (optional)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing processed prediction files (batch mode)')
    parser.add_argument('--original_toxic_file', type=str,
                        default='new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl',
                        help='Path to original toxic spans data file')
    parser.add_argument('--original_toxic_extended_file', type=str,
                        default='new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl',
                        help='Path to original toxic spans data file')
    parser.add_argument('--original_inconsistent_file', type=str,
                        default='new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl',
                        help='Path to original inconsistent spans data file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results (batch mode)')
    
    args = parser.parse_args()
    
    # Batch processing mode
    if args.results_dir:
        evaluate_all_results(
            args.results_dir,
            args.original_toxic_extended_file,
            args.original_toxic_file,
            args.original_inconsistent_file,
            args.output_dir
        )
    # Single file processing mode
    elif args.prediction_file and args.original_file and args.task:
        evaluate_single_file(
            args.prediction_file,
            args.original_file,
            args.task,
            args.output_file
        )
    else:
        parser.print_help()
        print("\nError: Either provide --results_dir for batch processing, "
              "or provide --prediction_file, --original_file, and --task for single file processing.")


if __name__ == '__main__':
    main()

