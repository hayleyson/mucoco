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
from typing import List, Dict
import warnings

# Import evaluation functions from evaluate_locate_nli.py
from new_module.evaluation.evaluate_locate.evaluate_locate_nli import (
    apply_ap,
    apply_precision,
    apply_recall,
    apply_f1,
    get_rr,
)

# Import GPT2 tokenizer for token-to-word conversion
from transformers import GPT2Tokenizer


def tokenize_by_whitespace(text: str) -> List[str]:
    """Tokenize text by whitespace, keeping internal apostrophes and hyphens."""
    return text.split()


def get_word2tok(tokens: List[int], words: List[str], tokenizer: GPT2Tokenizer, ws: str = " ") -> Dict[int, List[int]]:
    """
    Create a mapping from word indices to token indices.
    
    This function is based on the logic from evaluate_locate.py.
    It iteratively decodes tokens and matches them to words.
    
    Args:
        tokens: List of token IDs
        words: List of words (tokenized by whitespace)
        tokenizer: Tokenizer instance
        ws: Whitespace character used to join words (default: " ")
    
    Returns:
        Dictionary mapping word index to list of token indices
    """
    jl, jr, k = 0, 0, 0
    grouped_tokens = []
    
    if ws is not None:
        while jr <= len(tokens) and k < len(words):
            decoded = tokenizer.decode(tokens[jl:jr]).strip(' ')
            if decoded == words[k]:
                grouped_tokens.append(list(range(jl, jr)))
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1
    else:
        while jr <= len(tokens) and k < len(words):
            decoded = tokenizer.decode(tokens[jl:jr]).strip()
            if decoded == words[k]:
                grouped_tokens.append(list(range(jl, jr)))
                k += 1
                jl = jr
                jr += 1
            else:
                jr += 1
    
    word2tok = dict(zip(range(len(grouped_tokens)), grouped_tokens))
    return word2tok


def convert_gpt2_token_labels_to_word_labels(
    tokens: List[int],
    token_labels: List[float],
    text: str,
    tokenizer: GPT2Tokenizer
) -> List[float]:
    """
    Convert GPT2 token-level labels to word-level labels.
    
    Uses the get_word2tok function logic from evaluate_locate.py to create
    a mapping from words to tokens, then assigns max label value for each word.
    
    Args:
        tokens: List of GPT2 token IDs
        token_labels: List of token-level labels (can be continuous)
        text: The original text
        tokenizer: GPT2 tokenizer instance
    
    Returns:
        List of word-level labels (continuous values, may need binarization)
    """
    words = tokenize_by_whitespace(text)
    word_labels = [0.0] * len(words)
    
    # Create word-to-token mapping using the proven algorithm from evaluate_locate.py
    word2tok = get_word2tok(tokens, words, tokenizer, ws=" ")
    
    # Assign labels: for each word, use max label of its tokens
    for word_idx, token_indices in word2tok.items():
        if word_idx < len(word_labels) and token_indices:
            valid_indices = [i for i in token_indices if 0 <= i < len(token_labels)]
            if valid_indices:
                word_labels[word_idx] = max([token_labels[i] for i in valid_indices])
    
    return word_labels


def binarize_labels(labels: List[float], threshold: float = 0.5) -> List[int]:
    """Binarize continuous labels at the given threshold."""
    return [1 if label >= threshold else 0 for label in labels]


def calculate_exact_match(gold_binary: List[int], pred_binary: List[int]) -> int:
    """
    Calculate exact match: returns 1 if predictions exactly match gold labels, 0 otherwise.
    
    Args:
        gold_binary: List of gold binary labels
        pred_binary: List of predicted binary labels
    
    Returns:
        1 if exact match, 0 otherwise
    """
    # Ensure same length
    min_len = min(len(gold_binary), len(pred_binary))
    gold_binary = gold_binary[:min_len]
    pred_binary = pred_binary[:min_len]
    
    # If lengths differ, it's not an exact match
    if len(gold_binary) != len(pred_binary):
        return 0
    
    # Check if all labels match
    return 1 if gold_binary == pred_binary else 0


def load_gold_labels_toxic(original_file: str) -> List[Dict]:
    """
    Load gold labels for toxic span detection.
    Converts token-level labels to word-level labels.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            generation = data['generation']
            tokens = data['generation_gpt2_tokens']
            token_labels = data['generation_gpt2_token_labels']
            
            # Convert token labels to word labels
            word_labels_continuous = convert_gpt2_token_labels_to_word_labels(
                tokens, token_labels, generation, tokenizer
            )
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(word_labels_continuous, threshold=0.5)
            
            words = tokenize_by_whitespace(generation)
            
            gold_data.append({
                'index': len(gold_data),
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
            hypothesis_words = data.get('hypothesis_words', [])
            hypothesis_word_labels = data.get('hypothesis_word_labels', [])
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(hypothesis_word_labels, threshold=0.5)
            
            gold_data.append({
                'index': len(gold_data),
                'pairID': data.get('pairID', ''),
                'hypothesis_words': hypothesis_words,
                'word_labels_continuous': hypothesis_word_labels,
                'word_labels_binary': word_labels_binary
            })
    
    return gold_data


def load_predictions(prediction_file: str) -> List[Dict]:
    """Load predictions from post-processed results file."""
    predictions = []
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def evaluate_predictions(
    gold_data: List[Dict],
    predictions: List[Dict],
    task: str
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
    if task == 'toxic':
        gold_data = load_gold_labels_toxic(original_file)
    elif task == 'inconsistent':
        gold_data = load_gold_labels_inconsistent(original_file)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    print(f"Loaded {len(gold_data)} gold examples")
    
    # Load predictions
    print("Loading predictions...")
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
    metrics = evaluate_predictions(gold_data, predictions, task)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of examples: {metrics['num_examples']}")
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
        if 'toxic' in pred_file.name:
            task = 'toxic'
            original_file = original_toxic_file
        elif 'incon' in pred_file.name or 'inconsistent' in pred_file.name:
            task = 'inconsistent'
            original_file = original_inconsistent_file
        else:
            print(f"Warning: Could not determine task type for {pred_file.name}, skipping...")
            continue
        
        # Evaluate
        try:
            metrics = evaluate_single_file(
                str(pred_file),
                original_file,
                task
            )
            
            metrics['prediction_file'] = str(pred_file)
            metrics['task'] = task
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
        print(df_summary[['prediction_file', 'task', 'mean_precision', 'mean_recall', 'mean_f1', 'mean_ap', 'mean_rr', 'exact_match']].to_string(index=False))
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
    parser.add_argument('--task', type=str, choices=['toxic', 'inconsistent'], default=None,
                        help='Task type: toxic or inconsistent (single file mode)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save detailed results (optional)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing processed prediction files (batch mode)')
    parser.add_argument('--original_toxic_file', type=str,
                        default='new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl',
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

