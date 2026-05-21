#!/usr/bin/env python3
"""
Calculate recall separately for first and second half of input texts.
Recall = TP / (TP + FN) where:
- TP: predicted tokens that are actually toxic
- FN: actual toxic tokens that were not predicted
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def calculate_recall_for_half(word_labels, pred_indexes, start_idx, end_idx):
    """
    Calculate recall for a specific half of the text.
    
    Args:
        word_labels: List of ground truth labels (0 or 1)
        pred_indexes: List of predicted token indices
        start_idx: Start index (inclusive) for this half
        end_idx: End index (exclusive) for this half
    
    Returns:
        recall, tp, fn, total_positive
    """
    # Get ground truth toxic tokens in this half
    gt_toxic = set()
    for i in range(start_idx, min(end_idx, len(word_labels))):
        if word_labels[i] == 1:
            gt_toxic.add(i)
    
    # Get predicted tokens in this half
    pred_in_half = [idx for idx in pred_indexes if start_idx <= idx < end_idx]
    pred_set = set(pred_in_half)
    
    # Calculate TP and FN
    tp = len(pred_set & gt_toxic)  # Predicted and actually toxic
    fn = len(gt_toxic - pred_set)   # Actually toxic but not predicted
    total_positive = len(gt_toxic)  # Total actual toxic tokens
    
    # Calculate recall
    if total_positive == 0:
        recall = None  # No positive examples in this half
    else:
        recall = tp / total_positive if total_positive > 0 else 0.0
    
    return recall, tp, fn, total_positive

def analyze_file(filepath):
    """Analyze a single JSONL file and return recall statistics."""
    results = {
        'file': os.path.basename(filepath),
        'total_samples': 0,
        'first_half_recalls': [],
        'second_half_recalls': [],
        'first_half_tp': [],
        'first_half_fn': [],
        'first_half_total_pos': [],
        'second_half_tp': [],
        'second_half_fn': [],
        'second_half_total_pos': [],
        'samples_with_first_half_toxic': 0,
        'samples_with_second_half_toxic': 0,
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                results['total_samples'] += 1
                
                words = data.get('words', [])
                word_labels = data.get('word_labels', [])
                pred_indexes = data.get('roberta_token_pred_indexes', [])
                
                if not words or not word_labels:
                    continue
                
                # Calculate midpoint
                midpoint = len(words) // 2
                
                # Calculate recall for first half
                first_recall, first_tp, first_fn, first_total_pos = calculate_recall_for_half(
                    word_labels, pred_indexes, 0, midpoint
                )
                
                # Calculate recall for second half
                second_recall, second_tp, second_fn, second_total_pos = calculate_recall_for_half(
                    word_labels, pred_indexes, midpoint, len(words)
                )
                
                # Store results
                if first_total_pos > 0:
                    results['samples_with_first_half_toxic'] += 1
                    if first_recall is not None:
                        results['first_half_recalls'].append(first_recall)
                    results['first_half_tp'].append(first_tp)
                    results['first_half_fn'].append(first_fn)
                    results['first_half_total_pos'].append(first_total_pos)
                
                if second_total_pos > 0:
                    results['samples_with_second_half_toxic'] += 1
                    if second_recall is not None:
                        results['second_half_recalls'].append(second_recall)
                    results['second_half_tp'].append(second_tp)
                    results['second_half_fn'].append(second_fn)
                    results['second_half_total_pos'].append(second_total_pos)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {filepath}: {e}")
                continue
    
    return results

def main():
    directory = Path("/home/hyeryung/data/mucoco/new_module/locate/results/toxicspans_extended")
    
    # Find all JSONL files
    jsonl_files = sorted(directory.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in the directory.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files\n")
    print("=" * 100)
    
    # Analyze each file
    all_results = {}
    overall_stats = {
        'first_half_recalls': [],
        'second_half_recalls': [],
        'first_half_tp': [],
        'first_half_fn': [],
        'first_half_total_pos': [],
        'second_half_tp': [],
        'second_half_fn': [],
        'second_half_total_pos': [],
    }
    
    for filepath in jsonl_files:
        results = analyze_file(filepath)
        all_results[results['file']] = results
        
        # Aggregate for overall statistics
        overall_stats['first_half_recalls'].extend(results['first_half_recalls'])
        overall_stats['second_half_recalls'].extend(results['second_half_recalls'])
        overall_stats['first_half_tp'].extend(results['first_half_tp'])
        overall_stats['first_half_fn'].extend(results['first_half_fn'])
        overall_stats['first_half_total_pos'].extend(results['first_half_total_pos'])
        overall_stats['second_half_tp'].extend(results['second_half_tp'])
        overall_stats['second_half_fn'].extend(results['second_half_fn'])
        overall_stats['second_half_total_pos'].extend(results['second_half_total_pos'])
    
    # Print per-file statistics
    print("\nPER-FILE STATISTICS:")
    print("=" * 100)
    for filename, results in sorted(all_results.items()):
        print(f"\n{filename}:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Samples with toxic tokens in first half: {results['samples_with_first_half_toxic']}")
        print(f"  Samples with toxic tokens in second half: {results['samples_with_second_half_toxic']}")
        
        if results['first_half_recalls']:
            avg_recall = sum(results['first_half_recalls']) / len(results['first_half_recalls'])
            total_tp = sum(results['first_half_tp'])
            total_fn = sum(results['first_half_fn'])
            total_pos = sum(results['first_half_total_pos'])
            macro_recall = total_tp / total_pos if total_pos > 0 else 0.0
            
            print(f"\n  FIRST HALF:")
            print(f"    Micro-averaged Recall: {avg_recall:.4f}")
            print(f"    Macro-averaged Recall: {macro_recall:.4f}")
            print(f"    Total TP: {total_tp}, Total FN: {total_fn}, Total Positives: {total_pos}")
        else:
            print(f"\n  FIRST HALF: No toxic tokens found")
        
        if results['second_half_recalls']:
            avg_recall = sum(results['second_half_recalls']) / len(results['second_half_recalls'])
            total_tp = sum(results['second_half_tp'])
            total_fn = sum(results['second_half_fn'])
            total_pos = sum(results['second_half_total_pos'])
            macro_recall = total_tp / total_pos if total_pos > 0 else 0.0
            
            print(f"\n  SECOND HALF:")
            print(f"    Micro-averaged Recall: {avg_recall:.4f}")
            print(f"    Macro-averaged Recall: {macro_recall:.4f}")
            print(f"    Total TP: {total_tp}, Total FN: {total_fn}, Total Positives: {total_pos}")
        else:
            print(f"\n  SECOND HALF: No toxic tokens found")
        
        # Compare halves
        if results['first_half_recalls'] and results['second_half_recalls']:
            first_avg = sum(results['first_half_recalls']) / len(results['first_half_recalls'])
            second_avg = sum(results['second_half_recalls']) / len(results['second_half_recalls'])
            first_macro = sum(results['first_half_tp']) / sum(results['first_half_total_pos']) if sum(results['first_half_total_pos']) > 0 else 0
            second_macro = sum(results['second_half_tp']) / sum(results['second_half_total_pos']) if sum(results['second_half_total_pos']) > 0 else 0
            
            diff = first_macro - second_macro
            print(f"\n  COMPARISON (Macro-averaged):")
            print(f"    First half recall: {first_macro:.4f}")
            print(f"    Second half recall: {second_macro:.4f}")
            print(f"    Difference (First - Second): {diff:.4f}")
            if diff > 0.05:
                print(f"    ⚠️  First half recall is significantly higher")
            elif diff < -0.05:
                print(f"    ⚠️  Second half recall is significantly higher")
            else:
                print(f"    ✓  Recalls are similar")
    
    # Print overall statistics
    print("\n" + "=" * 100)
    print("OVERALL STATISTICS (across all files):")
    print("=" * 100)
    
    if overall_stats['first_half_recalls']:
        first_avg = sum(overall_stats['first_half_recalls']) / len(overall_stats['first_half_recalls'])
        first_total_tp = sum(overall_stats['first_half_tp'])
        first_total_fn = sum(overall_stats['first_half_fn'])
        first_total_pos = sum(overall_stats['first_half_total_pos'])
        first_macro = first_total_tp / first_total_pos if first_total_pos > 0 else 0.0
        
        print(f"\nFIRST HALF (across all files):")
        print(f"  Micro-averaged Recall: {first_avg:.4f}")
        print(f"  Macro-averaged Recall: {first_macro:.4f}")
        print(f"  Total TP: {first_total_tp}, Total FN: {first_total_fn}, Total Positives: {first_total_pos}")
    else:
        print(f"\nFIRST HALF: No toxic tokens found")
    
    if overall_stats['second_half_recalls']:
        second_avg = sum(overall_stats['second_half_recalls']) / len(overall_stats['second_half_recalls'])
        second_total_tp = sum(overall_stats['second_half_tp'])
        second_total_fn = sum(overall_stats['second_half_fn'])
        second_total_pos = sum(overall_stats['second_half_total_pos'])
        second_macro = second_total_tp / second_total_pos if second_total_pos > 0 else 0.0
        
        print(f"\nSECOND HALF (across all files):")
        print(f"  Micro-averaged Recall: {second_avg:.4f}")
        print(f"  Macro-averaged Recall: {second_macro:.4f}")
        print(f"  Total TP: {second_total_tp}, Total FN: {second_total_fn}, Total Positives: {second_total_pos}")
    else:
        print(f"\nSECOND HALF: No toxic tokens found")
    
    # Overall comparison
    if overall_stats['first_half_recalls'] and overall_stats['second_half_recalls']:
        first_macro = sum(overall_stats['first_half_tp']) / sum(overall_stats['first_half_total_pos']) if sum(overall_stats['first_half_total_pos']) > 0 else 0
        second_macro = sum(overall_stats['second_half_tp']) / sum(overall_stats['second_half_total_pos']) if sum(overall_stats['second_half_total_pos']) > 0 else 0
        diff = first_macro - second_macro
        
        print(f"\n" + "=" * 100)
        print("OVERALL COMPARISON (Macro-averaged):")
        print("=" * 100)
        print(f"First half recall:  {first_macro:.4f}")
        print(f"Second half recall: {second_macro:.4f}")
        print(f"Difference (First - Second): {diff:.4f}")
        
        if diff > 0.05:
            print(f"\n⚠️  WARNING: First half recall is significantly higher than second half.")
            print(f"   This suggests the model may have better recall for tokens in the")
            print(f"   former part of the input, possibly due to training on shorter excerpts.")
        elif diff < -0.05:
            print(f"\n⚠️  Second half recall is significantly higher than first half.")
        else:
            print(f"\n✓  Recalls are similar between first and second halves.")
            print(f"   No significant bias detected.")

if __name__ == "__main__":
    main()

