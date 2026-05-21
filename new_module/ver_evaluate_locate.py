

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
