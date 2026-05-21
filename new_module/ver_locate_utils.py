
def get_word2tok(row: pd.Series, tokenizer: AutoTokenizer) -> dict:
    """
    A function that take a list of words and a corresponding list of tokens 
    into a mapping between each word's index and its corresponding token indexes.
    @param row: A row from dataframe
    @return word2char: A dictionary with word's location index as keys and tuples of corresponding token location indexes as values.

    Example:
    row=pd.Series()
    row['words']=['wearing', 'games', 'and', 'holy', '****ing', 'shit', 'do', 'I', 'hate', 'horse', 'wearing', 'games.']
    row['tokens']=[86, 6648, 1830, 290, 11386, 25998, 278, 7510, 466, 314, 5465, 8223, 5762, 1830, 13]
    word2tok=get_word2tok(row)
    word2tok
    {0: [0, 1],
    1: [2],
    2: [3],
    ...
    10: [12],
    11: [13, 14]}
    """
    
    jl, jr, k = 0, 0, 0
    grouped_tokens = []
    tok2word=dict()
    while jr <= len(row['tokens'])+1 and k < len(row['words']):
        
        if tokenizer.decode(row['tokens'][jl:jr]).strip() == row['words'][k]:
            grouped_tokens.append(list(range(jl,jr)))
            for ix in range(jl,jr):
                tok2word[ix] = k
            k += 1
            jl = jr
            jr += 1
        else:
            jr += 1
    # word2tok = dict(zip(range(len(grouped_tokens)), grouped_tokens))
    # return word2tok
    return tok2word, grouped_tokens
