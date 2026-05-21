
from typing import List, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from scipy import stats
from transformers import GPT2Tokenizer

def apply_ap(row, binary_labels_col:str, pred_scores_col:str):

    if sum(row[binary_labels_col])==0:
        return np.nan
    else:
        return average_precision_score(row[binary_labels_col],row[pred_scores_col])
    
def apply_precision(row, binary_labels_col:str, binary_preds_col:str):

    return precision_score(row[binary_labels_col],row[binary_preds_col], zero_division=np.nan)

def apply_recall(row, binary_labels_col:str, binary_preds_col:str):

    return recall_score(row[binary_labels_col],row[binary_preds_col], zero_division=np.nan)

def apply_f1(row, binary_labels_col:str, binary_preds_col:str):

    return f1_score(row[binary_labels_col],row[binary_preds_col], zero_division=np.nan)

def rr(out, labels, k = 6): #implement mean reciprocal rank
    idx_array = stats.rankdata(-out, axis=-1, method='min')
    # print(idx_array)
    labels = np.where(labels==1)[0].astype(int)
    # print(labels)
    rank = np.take_along_axis(idx_array, labels, axis=-1)
    # print(rank)
    rr=1/rank.min() if rank.min() <= k else 0.
    return rr

def get_rr(row, binary_labels_col:str, pred_scores_col:str):
    """suffix should start with _"""
    if sum(row[binary_labels_col])==0:
        return np.nan
    else:
        return rr(np.array(row[pred_scores_col]),np.array(row[binary_labels_col]))


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


def get_word2tok(tokens: List[int], words: List[str], tokenizer, ws: str = None) -> Dict[int, List[int]]:
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



def tokenize_by_whitespace(text: str) -> List[str]:
    """Tokenize text by whitespace, keeping internal apostrophes and hyphens."""
    return text.split()


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

