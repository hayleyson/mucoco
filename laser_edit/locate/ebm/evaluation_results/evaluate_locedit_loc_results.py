"""
Evaluate locate results from new_locate_utils.py by comparing predictions with gold labels.

This script:
1. Loads gold labels from original data files
2. Loads predictions from new_locate_utils.py results
3. Converts token-level predictions to word-level predictions
4. Calculates metrics (precision, recall, F1, AP, RR, exact match) using functions from evaluate_locate_nli.py
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import re

from laser_edit.evaluation.locate_metrics_utils import (
    apply_ap,
    apply_precision,
    apply_recall,
    apply_f1,
    get_rr,
    get_word2tok,
    binarize_labels,
    tokenize_by_whitespace,
    calculate_exact_match,
)

# Import tokenizers for token-to-word conversion
from transformers import GPT2Tokenizer, AutoTokenizer


def parse_filename_metadata(filename: str) -> Tuple[str, str]:
    """
    Parse filename to extract model type and method (with max_num_tokens).
    
    Args:
        filename: The filename (e.g., "energy_model_gradient_norm_max_num_tokens_7.jsonl" 
                  or "classifier_attention_max_num_tokens_7.jsonl")
    
    Returns:
        Tuple of (model, method) where:
        - model: 'energy_model' or 'classifier'
        - method: 'gradient_norm_max_num_tokens_7' or 'attention_max_num_tokens_7' (includes max_num_tokens)
    """
    filename_lower = filename.lower()
    
    # Extract model type
    model = None
    if 'energy_model' in filename_lower or 'energymodel' in filename_lower:
        model = 'energy_model'
    elif 'classifier' in filename_lower:
        model = 'classifier'
    
    # Extract method base type
    method_base = None
    if 'gradient_norm' in filename_lower or 'grad_norm' in filename_lower:
        method_base = 'gradient_norm'
    elif 'attention' in filename_lower:
        method_base = 'attention'
    
    # Extract max_num_tokens
    max_num_tokens = None
    # Look for pattern: max_num_tokens_<number>
    match = re.search(r'max_num_tokens[_-]?(\d+)', filename_lower)
    if match:
        max_num_tokens = match.group(1)
    
    # Construct method with max_num_tokens if both are found
    if method_base and max_num_tokens:
        method = f"{method_base}_max_num_tokens_{max_num_tokens}"
    elif method_base:
        method = method_base
    else:
        method = 'unknown'
    
    # Default values if not found
    if model is None:
        model = 'unknown'
    
    return model, method


def load_gold_labels_inconsistent(original_file: str) -> List[Dict]:
    """
    Load gold labels for inconsistent span detection.
    Uses hypothesis_word_labels and binarizes at 0.5.
    """
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            hypothesis = data.get('hypothesis', '')
            hypothesis_words = data.get('hypothesis_words', [])
            hypothesis_word_labels = data.get('hypothesis_word_labels', [])
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(hypothesis_word_labels, threshold=0.5)
            
            gold_data.append({
                'index': len(gold_data),
                'pairID': data.get('pairID', ''),
                'hypothesis': hypothesis,
                'hypothesis_words': hypothesis_words,
                'word_labels_continuous': hypothesis_word_labels,
                'word_labels_binary': word_labels_binary
            })
    
    return gold_data

def load_gold_labels_toxic(original_file: str) -> List[Dict]:
    """
    Load gold labels for toxic span detection.
    Converts GPT2 token-level labels to word-level labels.
    """
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            generation = data['generation']
            words = data['generation_words']
            word_labels_continuous = data['generation_word_labels']
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(word_labels_continuous, threshold=0.5)
            
            gold_data.append({
                'index': len(gold_data),
                'generation': generation,
                'generation_words': words,
                'word_labels_continuous': word_labels_continuous,
                'word_labels_binary': word_labels_binary
            })
    
    return gold_data


def load_gold_labels_toxic_extended(original_file: str) -> List[Dict]:
    """
    Load gold labels for toxic span detection.
    Converts GPT2 token-level labels to word-level labels.
    """
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    gold_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            generation = data['text']
            words = data['words']
            word_labels_continuous = data['word_labels']
            
            # Binarize at 0.5
            word_labels_binary = binarize_labels(word_labels_continuous, threshold=0.5)
            
            gold_data.append({
                'index': len(gold_data),
                'generation': generation,
                'generation_words': words,
                'word_labels_continuous': word_labels_continuous,
                'word_labels_binary': word_labels_binary
            })
    
    return gold_data



def convert_roberta_token_predictions_to_word_level(
    text: str,
    token_pred_scores: List[float],
    token_pred_indexes: List[int],
    tokenizer: AutoTokenizer,
    words: List[str]
) -> tuple:
    """
    Convert RoBERTa token-level predictions to word-level predictions.
    
    Args:
        text: The original text (hypothesis for inconsistent task, generation for toxic task)
        token_pred_scores: List of token-level prediction scores (already filtered to hypothesis-only for NLI task)
        token_pred_indexes: List of token indices that were predicted (already relative to hypothesis-only for NLI task)
        tokenizer: RoBERTa tokenizer instance
        words: List of words (tokenized by whitespace)
    
    Returns:
        Tuple of (word_labels_binary, word_labels_scores):
        - word_labels_binary: Binary word-level predictions (1 if any token in word is predicted)
        - word_labels_scores: Word-level scores (max score of tokens in the word)
    """
    # Tokenize text directly (scores and indices are already filtered to hypothesis-only for NLI)
    tokenized = tokenizer(text, add_special_tokens=False)
    all_tokens = tokenized['input_ids']
    
    # Create word-to-token mapping
    word2tok = get_word2tok(all_tokens, words, tokenizer, ws=" ")
    
    # Create token-level binary predictions from token_pred_indexes
    token_pred_binary = [1 if i in token_pred_indexes else 0 for i in range(len(all_tokens))]
    
    # Initialize word-level predictions
    word_labels_binary = [0] * len(words)
    word_labels_scores = [0.0] * len(words)
    
    # For each word, check if any of its tokens are predicted
    for word_idx, token_indices in word2tok.items():
        if word_idx < len(words) and token_indices:
            # Check if any token in this word is predicted
            word_has_prediction = any(token_pred_binary[tok_idx] == 1 for tok_idx in token_indices if tok_idx < len(token_pred_binary))
            if word_has_prediction:
                word_labels_binary[word_idx] = 1
            
            # Get max score for tokens in this word
            word_token_scores = []
            for tok_idx in token_indices:
                if tok_idx < len(token_pred_scores):
                    word_token_scores.append(token_pred_scores[tok_idx])
            
            if word_token_scores:
                word_labels_scores[word_idx] = max(word_token_scores)
    
    return word_labels_binary, word_labels_scores


def load_predictions_from_locate_results(
    prediction_file: str, 
    task: str, 
    tokenizer_name: str = 'roberta-base',
    gold_data: List[Dict] = None
) -> List[Dict]:
    """
    Load predictions from new_locate_utils.py results file.
    Converts token-level predictions to word-level predictions.
    
    Args:
        prediction_file: Path to prediction file
        task: 'toxic' or 'inconsistent'
        tokenizer_name: Name of the tokenizer to use (default: 'roberta-base')
        gold_data: Optional gold data for getting original text and words
    
    Returns:
        List of prediction dictionaries with word-level labels
    """
    predictions = []
    
    # Load RoBERTa tokenizer once
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except:
        # Fallback
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)

            if task == 'toxic_extended':
                prompt = ""
                generations = [data]
            else:
                prompt = data.get('prompt', {})
                generations = data.get('generations', [])
            
            if not generations:
                continue
            
            # For now, take the first generation (assuming one per prompt)
            gen = generations[0]
            masked_text = gen.get('text', '')
            token_pred_scores = gen.get('roberta_token_pred_scores', [])
            token_pred_indexes = gen.get('roberta_token_pred_indexes', [])
            
            # Get original text and words from gold data for proper alignment
            # Note: token_pred_scores and token_pred_indexes are already filtered to hypothesis-only for NLI task
            if gold_data and idx < len(gold_data):
                if task == 'inconsistent':
                    # Get original hypothesis text and words from gold data
                    original_text = gold_data[idx].get('hypothesis', '')
                    words = gold_data[idx].get('hypothesis_words', [])
                    if not words:
                        words = tokenize_by_whitespace(original_text)
                    
                elif (task == 'toxic') or (task == 'toxic_extended'):
                    # Get original generation text and words from gold data
                    original_text = gold_data[idx].get('generation', '')
                    words = gold_data[idx].get('generation_words', [])
                    if not words:
                        words = tokenize_by_whitespace(original_text)
                        
                else:
                    raise ValueError(f"Unknown task: {task}")
            else:
                # Fallback: try to extract words from masked text
                text_for_words = masked_text.replace('<mask>', '').replace('<mask><mask>', '').strip()
                text_for_words = ' '.join(text_for_words.split())
                words = tokenize_by_whitespace(text_for_words) if text_for_words else []
                if not words:
                    words = tokenize_by_whitespace(masked_text)
                original_text = masked_text
            
            # Convert token-level predictions to word-level predictions
            # token_pred_scores and token_pred_indexes are already filtered to hypothesis-only for NLI task
            word_labels_binary, word_labels_scores = convert_roberta_token_predictions_to_word_level(
                original_text, token_pred_scores, token_pred_indexes, tokenizer, words
            )
            
            predictions.append({
                'prompt_text': prompt.get('text', '') if isinstance(prompt, dict) else prompt,
                'text': original_text,
                'words': words,
                'word_labels': word_labels_binary,  # Binary word-level predictions
                'word_labels_scores': word_labels_scores,  # Word-level scores for AP/RR
                'token_pred_scores': token_pred_scores,
                'token_pred_indexes': token_pred_indexes
            })
    
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
        pred_binary = pred['word_labels']  # Already word-level binary from token_pred_indexes
        pred_scores = pred['word_labels_scores']  # Word-level scores from token_pred_scores
        
        # Ensure same length
        min_len = min(len(gold_binary), len(pred_binary), len(pred_scores))
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
    output_file: str = None,
    tokenizer_name: str = 'roberta-base'
):
    """
    Evaluate a single prediction file against gold labels.
    
    Args:
        prediction_file: Path to locate results file from new_locate_utils.py
        original_file: Path to original data file with gold labels
        task: 'toxic' or 'inconsistent'
        output_file: Optional path to save detailed results
        tokenizer_name: Name of the tokenizer to use (default: 'roberta-base')
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
    else:
        raise ValueError(f"Unknown task: {task}")
    
    print(f"Loaded {len(gold_data)} gold examples")
    
    # Load predictions
    print("Loading predictions...")
    # Pass gold_data to get original text (needed for both tasks to ensure proper token alignment)
    predictions = load_predictions_from_locate_results(
        prediction_file, task, tokenizer_name, gold_data
    )
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
    original_toxic_extended_file: str,
    original_toxic_file: str,
    original_inconsistent_file: str,
    output_dir: str = None,
    tokenizer_name: str = 'roberta-base'
):
    """
    Evaluate all prediction files in the results directory recursively.
    
    The results directory should have subdirectories:
    - toxicspans/ - for toxic span results
    - inconsistentspans/ - for inconsistent span results
    - inconsistentspans_saehee/ - also for inconsistent spans
    
    Recursively finds all .jsonl files in subdirectories and evaluates them.
    """
    results_path = Path(results_dir)
    all_metrics = []
    print(results_path)
    print(results_path.rglob('*.jsonl'))
    
    # Recursively find all .jsonl files (excluding .time files)
    all_jsonl_files = list(results_path.rglob('*.jsonl'))
    print(all_jsonl_files)
    prediction_files = [f for f in all_jsonl_files if not f.name.endswith('.time')]
    
    print(f"Found {len(prediction_files)} prediction files to evaluate\n")
    
    for pred_file in prediction_files:
        # Determine task type from parent directory name
        try:
            # Get the parent directory name directly - this is more reliable
            parent_dir_name = pred_file.parent.name
            print(parent_dir_name)
            
            # Check parent directory name for task type
            task = None
            original_file = None
            subdirectory = None
            
            if 'toxicspans_extended' == parent_dir_name.lower():
                task = 'toxic_extended'
                original_file = original_toxic_extended_file
                subdirectory = parent_dir_name
            elif 'toxicspans' == parent_dir_name.lower():
                task = 'toxic'
                original_file = original_toxic_file
                subdirectory = parent_dir_name
            elif 'inconsistent' in parent_dir_name.lower() or 'incon' in parent_dir_name.lower():
                
                task = 'inconsistent'
                original_file = original_inconsistent_file
                subdirectory = parent_dir_name
            else:
                # Try to get relative path as fallback
                try:
                    results_path_resolved = results_path.resolve()
                    pred_file_resolved = pred_file.resolve()
                    relative_path = pred_file_resolved.relative_to(results_path_resolved)
                    parent_dirs = relative_path.parts[:-1] if len(relative_path.parts) > 1 else []
                    
                    # Check parent directories
                    for parent_dir in parent_dirs:
                        if 'toxicspans_extended' == parent_dir.lower():
                            task = 'toxic_extended'
                            original_file = original_toxic_extended_file
                            subdirectory = parent_dir
                            break
                        elif 'toxicspans' == parent_dir.lower():
                            task = 'toxic'
                            original_file = original_toxic_file
                            subdirectory = parent_dir
                            break
                        elif 'inconsistent' in parent_dir.lower() or 'incon' in parent_dir.lower():
                            task = 'inconsistent'
                            original_file = original_inconsistent_file
                            subdirectory = parent_dir
                            break
                except (ValueError, AttributeError):
                    pass
            
            if task is None:
                print(f"Warning: Could not determine task type for {pred_file}, skipping...")
                continue
            
            # Try to read execution time from corresponding .time file
            execution_seconds = None
            time_file = pred_file.parent / f"{pred_file.stem}.time"
            if time_file.exists():
                try:
                    with open(time_file, 'r') as f:
                        time_content = f.read().strip()
                        # Try to parse as float
                        execution_seconds = float(time_content)
                except (ValueError, IOError) as e:
                    print(f"Warning: Could not read execution time from {time_file}: {e}")
            
            # Parse filename to extract model and method
            model, method = parse_filename_metadata(pred_file.name)
            
            print(f"\nProcessing: {subdirectory}/{pred_file.name}")
            try:
                metrics = evaluate_single_file(
                    str(pred_file),
                    original_file,
                    task,
                    tokenizer_name=tokenizer_name
                )
                
                metrics['prediction_file'] = str(pred_file)
                
                metrics['task'] = task
                metrics['model'] = model
                metrics['method'] = method
                if execution_seconds is not None:
                    metrics['execution_seconds'] = execution_seconds
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error evaluating {pred_file.name}: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_metrics:
        print("\n" + "="*80)
        print("SUMMARY OF ALL RESULTS")
        print("="*80)
        df_summary = pd.DataFrame(all_metrics)
        
        # Create a cleaner display with task, model, method
        display_cols = ['task', 'model', 'method', 'mean_precision', 'mean_recall', 'mean_f1', 'mean_ap', 'mean_rr', 'exact_match', 'num_examples', 'execution_seconds']
        display_cols = [c for c in display_cols if c in df_summary.columns]
        
        print(df_summary[display_cols].to_string(index=False))
        print("="*80 + "\n")
        
        # Save summary if output_dir is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            summary_file = output_path / 'evaluation_summary.csv'
            # Save with all columns including full paths
            df_summary.to_csv(summary_file, index=False)
            print(f"Summary saved to {summary_file}\n")
            print(f"Total files evaluated: {len(all_metrics)}")
    else:
        print("No results to summarize.")


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
                        help='Directory containing locate results (batch mode). Should have subdirectories: toxicspans/, inconsistentspans/, etc.')
    parser.add_argument('--original_toxic_extended_file', type=str,
                        default='laser_edit/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl',
                        help='Path to original toxic spans extended data file')
    parser.add_argument('--original_toxic_file', type=str,
                        default='laser_edit/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl',
                        help='Path to original toxic spans data file')
    parser.add_argument('--original_inconsistent_file', type=str,
                        default='laser_edit/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl',
                        help='Path to original inconsistent spans data file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results (batch mode)')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base',
                        help='Tokenizer name to use for RoBERTa tokenization (default: roberta-base)')
    
    args = parser.parse_args()
    
    # Batch processing mode
    if args.results_dir:
        evaluate_all_results(
            args.results_dir,
            args.original_toxic_extended_file,
            args.original_toxic_file,
            args.original_inconsistent_file,
            args.output_dir,
            args.tokenizer_name
        )
    # Single file processing mode
    elif args.prediction_file and args.original_file and args.task:
        evaluate_single_file(
            args.prediction_file,
            args.original_file,
            args.task,
            args.output_file,
            args.tokenizer_name
        )
    else:
        parser.print_help()
        print("\nError: Either provide --results_dir for batch processing, "
              "or provide --prediction_file, --original_file, and --task for single file processing.")


if __name__ == '__main__':
    main()

