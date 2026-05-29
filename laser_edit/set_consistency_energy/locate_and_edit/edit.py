def mask_text(text, mask_ixes, max_mask_cnt_per_span, dataset='convqa'):
    tmp = ''
    if dataset == 'convqa':
        qa_pairs = text.split('.')
        for i, qa_pair in enumerate(qa_pairs):
            if i in mask_ixes:
                q, a = qa_pair.split('?')
                tmp += q + '? The answer is ' + '<mask>' * max_mask_cnt_per_span + '.'
            else:
                tmp += qa_pair + '.'
            if i != len(qa_pairs) - 1:
                tmp += ' '
    return tmp