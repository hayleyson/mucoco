#!/usr/bin/env python3
"""
Analyze roberta_token_pred_indexes across all JSONL files to check if tokens
are only identified in the former part of the input text.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_file(filepath):
    """Analyze a single JSONL file and return statistics."""
    results = {
        'file': os.path.basename(filepath),
        'total_samples': 0,
        'all_indexes': [],
        'relative_positions': [],
        'text_lengths': [],
        'num_predicted_tokens': []
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                results['total_samples'] += 1
                
                # Get roberta_token_pred_indexes
                pred_indexes = data.get('roberta_token_pred_indexes', [])
                if pred_indexes:
                    results['all_indexes'].extend(pred_indexes)
                    results['num_predicted_tokens'].append(len(pred_indexes))
                    
                    # Calculate relative positions
                    words = data.get('words', [])
                    text_length = len(words)
                    results['text_lengths'].append(text_length)
                    
                    if text_length > 0:
                        for idx in pred_indexes:
                            # Normalize by text length (0.0 = start, 1.0 = end)
                            relative_pos = idx / text_length
                            results['relative_positions'].append(relative_pos)
                else:
                    # No predictions for this sample
                    words = data.get('words', [])
                    results['text_lengths'].append(len(words))
                    results['num_predicted_tokens'].append(0)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {filepath}: {e}")
                continue
    
    return results

def main():
    directory = Path("/home/hyeryung/data/mucoco/laser_edit/locate/results/toxicspans_extended")
    
    # Find all JSONL files
    jsonl_files = sorted(directory.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in the directory.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files\n")
    print("=" * 80)
    
    # Analyze each file
    all_results = {}
    overall_stats = {
        'all_indexes': [],
        'relative_positions': [],
        'text_lengths': [],
        'num_predicted_tokens': []
    }
    
    for filepath in jsonl_files:
        results = analyze_file(filepath)
        all_results[results['file']] = results
        
        # Aggregate for overall statistics
        overall_stats['all_indexes'].extend(results['all_indexes'])
        overall_stats['relative_positions'].extend(results['relative_positions'])
        overall_stats['text_lengths'].extend(results['text_lengths'])
        overall_stats['num_predicted_tokens'].extend(results['num_predicted_tokens'])
    
    # Print per-file statistics
    print("\nPER-FILE STATISTICS:")
    print("=" * 80)
    for filename, results in sorted(all_results.items()):
        print(f"\n{filename}:")
        print(f"  Total samples: {results['total_samples']}")
        
        if results['all_indexes']:
            avg_index = statistics.mean(results['all_indexes'])
            median_index = statistics.median(results['all_indexes'])
            min_index = min(results['all_indexes'])
            max_index = max(results['all_indexes'])
            
            print(f"  Token indexes:")
            print(f"    Average: {avg_index:.2f}")
            print(f"    Median: {median_index:.2f}")
            print(f"    Min: {min_index}")
            print(f"    Max: {max_index}")
            
            if results['relative_positions']:
                avg_rel_pos = statistics.mean(results['relative_positions'])
                median_rel_pos = statistics.median(results['relative_positions'])
                print(f"  Relative positions (0.0=start, 1.0=end):")
                print(f"    Average: {avg_rel_pos:.3f} ({avg_rel_pos*100:.1f}% through text)")
                print(f"    Median: {median_rel_pos:.3f} ({median_rel_pos*100:.1f}% through text)")
            
            avg_text_len = statistics.mean(results['text_lengths']) if results['text_lengths'] else 0
            print(f"  Average text length: {avg_text_len:.1f} words")
            print(f"  Average predicted tokens per sample: {statistics.mean(results['num_predicted_tokens']):.2f}")
        else:
            print(f"  No predicted tokens found")
    
    # Print overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS (across all files):")
    print("=" * 80)
    
    if overall_stats['all_indexes']:
        print(f"\nTotal predicted token indexes: {len(overall_stats['all_indexes'])}")
        print(f"Average token index: {statistics.mean(overall_stats['all_indexes']):.2f}")
        print(f"Median token index: {statistics.median(overall_stats['all_indexes']):.2f}")
        print(f"Min token index: {min(overall_stats['all_indexes'])}")
        print(f"Max token index: {max(overall_stats['all_indexes'])}")
        
        if overall_stats['relative_positions']:
            print(f"\nRelative positions (0.0=start, 1.0=end):")
            avg_rel_pos = statistics.mean(overall_stats['relative_positions'])
            median_rel_pos = statistics.median(overall_stats['relative_positions'])
            std_rel_pos = statistics.stdev(overall_stats['relative_positions']) if len(overall_stats['relative_positions']) > 1 else 0
            
            print(f"  Average: {avg_rel_pos:.3f} ({avg_rel_pos*100:.1f}% through text)")
            print(f"  Median: {median_rel_pos:.3f} ({median_rel_pos*100:.1f}% through text)")
            print(f"  Std Dev: {std_rel_pos:.3f}")
            
            # Count how many are in first half vs second half
            first_half = sum(1 for pos in overall_stats['relative_positions'] if pos < 0.5)
            second_half = sum(1 for pos in overall_stats['relative_positions'] if pos >= 0.5)
            total = len(overall_stats['relative_positions'])
            
            print(f"\nDistribution:")
            print(f"  First half (0.0-0.5): {first_half} ({first_half/total*100:.1f}%)")
            print(f"  Second half (0.5-1.0): {second_half} ({second_half/total*100:.1f}%)")
            
            # First quarter vs rest
            first_quarter = sum(1 for pos in overall_stats['relative_positions'] if pos < 0.25)
            print(f"  First quarter (0.0-0.25): {first_quarter} ({first_quarter/total*100:.1f}%)")
        
        if overall_stats['text_lengths']:
            avg_text_len = statistics.mean(overall_stats['text_lengths'])
            print(f"\nAverage text length: {avg_text_len:.1f} words")
            print(f"Average predicted tokens per sample: {statistics.mean(overall_stats['num_predicted_tokens']):.2f}")
        
        print("\n" + "=" * 80)
        print("CONCLUSION:")
        print("=" * 80)
        if overall_stats['relative_positions']:
            avg_rel_pos = statistics.mean(overall_stats['relative_positions'])
            if avg_rel_pos < 0.3:
                print("⚠️  WARNING: Average relative position is in the first 30% of text.")
                print("   This suggests the model may be biased toward identifying tokens")
                print("   in the former part of the input, possibly due to training on")
                print("   shorter excerpts while test inputs are longer.")
            elif avg_rel_pos < 0.5:
                print("⚠️  CAUTION: Average relative position is in the first half of text.")
                print("   There may be some bias toward the former part of the input.")
            else:
                print("✓  Average relative position is in the second half of text.")
                print("   No strong bias toward the former part detected.")
    else:
        print("No predicted tokens found in any file.")

if __name__ == "__main__":
    main()

