import json
import re
from typing import List, Tuple
from pathlib import Path
import warnings
import shutil


def tokenize_by_whitespace(text: str) -> List[str]:
    """
    Tokenize text by whitespace, keeping internal apostrophes and hyphens.
    This matches the tokenization used in the prompts.
    """
    # Split on whitespace but preserve the words
    words = text.split()
    return words


def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for matching by removing extra whitespace and converting to lowercase.
    """
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    return text.lower().strip()


def find_span_in_text(span_text: str, text: str, words: List[str]) -> List[int]:
    """
    Find all occurrences of a span in the text and return the word indices.
    
    Args:
        span_text: The span text to find
        text: The full text
        words: List of words (tokenized)
    
    Returns:
        List of word indices where the span starts
    """
    word_indices = []
    
    # Normalize for matching
    span_normalized = normalize_text_for_matching(span_text)
    text_normalized = normalize_text_for_matching(text)
    
    # First, try exact match in normalized text
    if span_normalized in text_normalized:
        # Find all occurrences
        start = 0
        while True:
            idx = text_normalized.find(span_normalized, start)
            if idx == -1:
                break
            
            # Map character position to word index
            # Build character positions for each word
            char_positions = []
            char_pos = 0
            for word in words:
                char_positions.append((char_pos, char_pos + len(word)))
                char_pos += len(word) + 1  # +1 for space
            
            # Find which word contains this character position
            for word_idx, (start_pos, end_pos) in enumerate(char_positions):
                if start_pos <= idx < end_pos:
                    word_indices.append(word_idx)
                    break
            
            start = idx + 1
    
    # If no match found, try matching by words
    if not word_indices:
        span_words = tokenize_by_whitespace(span_text)
        span_words_normalized = [normalize_text_for_matching(w) for w in span_words]
        words_normalized = [normalize_text_for_matching(w) for w in words]
        
        if len(span_words_normalized) > 0:
            # Try to find consecutive occurrence of span words
            for i in range(len(words_normalized) - len(span_words_normalized) + 1):
                window = words_normalized[i:i+len(span_words_normalized)]
                if window == span_words_normalized:
                    word_indices.append(i)
                    # Only take first match to avoid duplicates
                    break
    
    return word_indices


def parse_llm_span(span_entry: str, text: str, words: List[str]) -> List[int]:
    """
    Parse a span entry from LLM output.
    
    Formats:
    - "text" (just the text)
    - "text, index" (text with word index)
    - "text, index1 index2" (text with multiple indices)
    
    Returns:
        List of word indices where this span appears
    """
    span_entry = span_entry.strip()
    
    # Check if it's in "text, index" format
    if ',' in span_entry:
        # Try to split: look for pattern "text, numbers"
        # Use regex to find the pattern
        match = re.match(r'^(.+?),\s*([0-9\s]+)$', span_entry)
        if match:
            span_text = match.group(1).strip()
            indices_str = match.group(2).strip()
            
            # Parse indices (can be single or multiple)
            indices = []
            for idx_str in indices_str.split():
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(words):
                        indices.append(idx)
                except ValueError:
                    pass
            
            if indices:
                return indices
        else:
            # Couldn't parse as "text, index", treat entire thing as text
            span_text = span_entry
    else:
        span_text = span_entry
    
    # Find span in text
    return find_span_in_text(span_text, text, words)


def create_word_labels_from_spans(spans: List, text: str, words: List[str]) -> List[int]:
    """
    Create binary word-level labels from LLM span predictions.
    
    Args:
        spans: List of span entries from LLM output (can be strings or dicts)
        text: The full text
        words: List of words (tokenized)
    
    Returns:
        Binary list where 1 indicates the word is part of a predicted span
    """
    labels = [0] * len(words)
    
    for span_entry in spans:
        # Handle dict format: {"span": "text", "word_index": [indices]}
        if isinstance(span_entry, dict):
            span_text = span_entry.get('span', '')
            word_indices = span_entry.get('word_index', [])
            
            if not span_text and not word_indices:
                continue
            
            # If word_index is provided, use it directly
            if word_indices:
                span_words = tokenize_by_whitespace(span_text) if span_text else []
                span_length = len(span_words) if span_words else 1
                
                for start_idx in word_indices:
                    if isinstance(start_idx, int) and 0 <= start_idx < len(words):
                        # Mark words from start_idx to start_idx + span_length
                        for i in range(span_length):
                            if start_idx + i < len(labels):
                                labels[start_idx + i] = 1
            else:
                # Fall back to text matching
                span_words = tokenize_by_whitespace(span_text)
                span_length = len(span_words)
                word_indices = find_span_in_text(span_text, text, words)
                
                for start_idx in word_indices:
                    for i in range(span_length):
                        if start_idx + i < len(labels):
                            labels[start_idx + i] = 1
        # Handle string format
        elif isinstance(span_entry, str):
            # Extract span text (before comma if present)
            if ',' in span_entry:
                # Try to extract text part
                match = re.match(r'^(.+?),\s*[0-9\s]+$', span_entry)
                if match:
                    span_text = match.group(1).strip()
                else:
                    span_text = span_entry.split(',')[0].strip()
            else:
                span_text = span_entry.strip()
            
            span_words = tokenize_by_whitespace(span_text)
            span_length = len(span_words)
            
            # Get word indices where this span appears
            word_indices = parse_llm_span(span_entry, text, words)
            
            # Mark all words in the span
            for start_idx in word_indices:
                # Mark words from start_idx to start_idx + span_length
                for i in range(span_length):
                    if start_idx + i < len(labels):
                        labels[start_idx + i] = 1
        else:
            # Unknown format, skip
            continue
    
    return labels


def process_toxic_spans(result_file: str, original_file: str, output_file: str):
    """
    Process toxic span detection results.
    """
    # Read original data
    original_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line))
    
    # Read LLM results
    llm_results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Results are JSON strings that are themselves quoted
            # Try parsing directly first (JSON can handle quoted strings)
            try:
                result = json.loads(line)
                # If result is a string, parse it again as JSON
                if isinstance(result, str):
                    llm_results.append(json.loads(result))
                else:
                    llm_results.append(result)
            except json.JSONDecodeError:
                # If parsing fails, assume that the model did not find any spans
                warnings.warn(f"Parsing failed for {line}. Will assume that the model did not find any spans.")
                llm_results.append({"spans": []})
    
    # Process each example
    output_data = []
    for i, (orig, result) in enumerate(zip(original_data, llm_results)):
        generation = orig['generation']
        words = tokenize_by_whitespace(generation)
        
        # Get spans from LLM result
        spans = result.get('spans', [])
        
        # Create binary labels
        word_labels = create_word_labels_from_spans(spans, generation, words)
        
        # Create output entry
        output_entry = {
            'index': i,
            'prompt': orig.get('prompt', ''),
            'generation': generation,
            'generation_words': words,
            'llm_spans': spans,
            'word_labels': word_labels
        }
        output_data.append(output_entry)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(output_data)} examples. Output saved to {output_file}")


def process_toxic_spans_extended(result_file: str, original_file: str, output_file: str):
    
    
    # Read original data
    original_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line))
    
    # Read LLM results
    llm_results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Results are JSON strings that are themselves quoted
            # Try parsing directly first (JSON can handle quoted strings)
            try:
                result = json.loads(line)
                # If result is a string, parse it again as JSON
                if isinstance(result, str):
                    llm_results.append(json.loads(result))
                else:
                    llm_results.append(result)
            except json.JSONDecodeError:
                # If parsing fails, assume that the model did not find any spans
                warnings.warn(f"Parsing failed for {line}. Will assume that the model did not find any spans.")
                llm_results.append({"spans": []})
    
    # Process each example
    output_data = []
    for i, (orig, result) in enumerate(zip(original_data, llm_results)):
        generation = orig['text']
        words = tokenize_by_whitespace(generation)
        
        # Get spans from LLM result
        spans = result.get('spans', [])
        
        # Create binary labels
        word_labels = create_word_labels_from_spans(spans, generation, words)
        
        # Create output entry
        output_entry = {
            'index': i,
            'text': generation,
            'words': words,
            'llm_spans': spans,
            'word_labels': word_labels
        }
        output_data.append(output_entry)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(output_data)} examples. Output saved to {output_file}")


def process_inconsistent_spans(result_file: str, original_file: str, output_file: str):
    """
    Process inconsistent span detection results.
    """
    # Read original data
    original_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line))
    
    # Read LLM results
    llm_results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Results are JSON strings that are themselves quoted
            # Try parsing directly first (JSON can handle quoted strings)
            try:
                result = json.loads(line)
                # If result is a string, parse it again as JSON
                if isinstance(result, str):
                    llm_results.append(json.loads(result))
                else:
                    llm_results.append(result)
            except json.JSONDecodeError:
                # If parsing fails, assume that the model did not find any spans
                llm_results.append({"spans": []})
                warnings.warn(f"Parsing failed for {line}. Will assume that the model did not find any spans.")

    
    # Process each example
    output_data = []
    for i, (orig, result) in enumerate(zip(original_data, llm_results)):
        # For inconsistent spans, we work with hypothesis
        hypothesis_words = orig.get('hypothesis_words', [])
        hypothesis_text = ' '.join(hypothesis_words)
        
        # Get spans from LLM result
        spans = result.get('spans', [])
        
        # Create binary labels
        word_labels = create_word_labels_from_spans(spans, hypothesis_text, hypothesis_words)
        
        # Create output entry
        output_entry = {
            'index': i,
            'pairID': orig.get('pairID', ''),
            'premise': orig.get('premise', ''),
            'hypothesis': orig.get('hypothesis', ''),
            'hypothesis_words': hypothesis_words,
            'llm_spans': spans,
            'word_labels': word_labels
        }
        output_data.append(output_entry)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(output_data)} examples. Output saved to {output_file}")


def process_bbm(result_file: str, output_file: str):
    """
    Process BIG-Bench-Mistake results by simply loading the result file,
    parsing it, and saving it as jsonl.
    """
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Results might be JSON strings or quoted strings containing JSON
                data = json.loads(line)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        pass
                results.append(data)
            except json.JSONDecodeError:
                continue
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(results)} examples. Output saved to {output_file}")


def process_all_results(results_dir: str, original_toxic_file: str, original_inconsistent_file: str, output_dir: str):
    """
    Process all result files in the results directory.
    
    Args:
        results_dir: Directory containing LLM result files
        original_toxic_file: Path to original toxic spans data file
        original_inconsistent_file: Path to original inconsistent spans data file
        output_dir: Directory to save processed outputs
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all result files
    result_files = list(results_path.glob('*.jsonl'))
    result_files = [f for f in result_files if not f.name.endswith('.time')]
    
    for result_file in result_files:
        # Determine task type from filename
        if 'toxicspans_extended' in result_file.name:
            task = 'toxic_extended'
            original_file = original_toxic_extended_file
        elif 'toxic' in result_file.name:
            task = 'toxic'
            original_file = original_toxic_file
        elif 'incon' in result_file.name or 'inconsistent' in result_file.name:
            task = 'inconsistent'
            original_file = original_inconsistent_file
        elif 'logical_deduction' in result_file.name:
            task = 'logical_deduction'
            original_file = None # Not needed for simple BBM pass-through
        elif 'tracking_shuffled_objects' in result_file.name:
            task = 'tracking_shuffled_objects'
            original_file = None # Not needed for simple BBM pass-through
        else:
            print(f"Warning: Could not determine task type for {result_file.name}, skipping...")
            continue
        
        # Create output filename
        output_file = output_path / f"{result_file.stem}_processed.jsonl"
        
        print(f"Processing {result_file.name}...")
        # try:
        if task == 'toxic_extended':
            process_toxic_spans_extended(str(result_file), original_file, str(output_file))
        elif task == 'toxic':
            process_toxic_spans(str(result_file), original_file, str(output_file))
        elif task == 'inconsistent':
            process_inconsistent_spans(str(result_file), original_file, str(output_file))
        else:
            # For BBM tasks (logical_deduction, tracking_shuffled_objects)
            process_bbm(str(result_file), str(output_file))
        # except Exception as e:
            # print(f"Error processing {result_file.name}: {e}")
    
    # Copy all .time files from results directory to output directory
    time_files = list(results_path.glob('*.time'))
    for time_file in time_files:
        dest_file = output_path / time_file.name
        shutil.copy2(time_file, dest_file)
        print(f"Copied {time_file.name} to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert LLM span predictions to word-level binary labels')
    parser.add_argument('--result_file', type=str, default=None,
                        help='Path to LLM result file (.jsonl). If not provided, processes all files in results_dir.')
    parser.add_argument('--original_file', type=str, default=None,
                        help='Path to original data file (.jsonl). Required if result_file is provided.')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to output file (.jsonl). Required if result_file is provided.')
    parser.add_argument('--task', type=str, choices=['toxic', 'inconsistent', 'logical_deduction', 'tracking_shuffled_objects'], default=None,
                        help='Task type: toxic or inconsistent or logical_deduction or tracking_shuffled_objects. Required if result_file is provided.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing LLM result files. If provided, processes all files.')
    parser.add_argument('--original_toxic_file', type=str, 
                        default='new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_115_locate_labels.jsonl',
                        help='Path to original toxic spans data file')
    parser.add_argument('--original_toxic_extended_file', type=str, 
                        default='new_module/data/locate/toxicspans/realtoxicityprompts_gpt2_gen_extended_locate_labels.jsonl',
                        help='Path to original toxic spans data file')
    parser.add_argument('--original_inconsistent_file', type=str,
                        default='new_module/data/locate/inconsistentspans/nli_contra_300_locate_labels_final.jsonl',
                        help='Path to original inconsistent spans data file')
    parser.add_argument('--output_dir', type=str, default='new_module/llm_experiments/locate_with_llm/processed_results',
                        help='Directory to save processed outputs (used when processing all files)')
    
    args = parser.parse_args()
    
    # Batch processing mode
    if args.results_dir:
        process_all_results(
            args.results_dir,
            args.original_toxic_extended_file,
            args.original_toxic_file,
            args.original_inconsistent_file,
            args.output_dir
        )
    # Single file processing mode
    elif args.result_file and args.original_file and args.output_file and args.task:
        if args.task == 'toxic_extended':
            process_toxic_spans_extended(args.result_file, args.original_file, args.output_file)
        if args.task == 'toxic':
            process_toxic_spans(args.result_file, args.original_file, args.output_file)
        elif args.task == 'inconsistent':
            process_inconsistent_spans(args.result_file, args.original_file, args.output_file)
        elif args.task in ['logical_deduction', 'tracking_shuffled_objects']:
            process_bbm(args.result_file, args.output_file)
    else:
        parser.print_help()
        print("\nError: Either provide --results_dir for batch processing, or provide --result_file, --original_file, --output_file, and --task for single file processing.")


if __name__ == '__main__':
    main()

