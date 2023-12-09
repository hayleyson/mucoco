import pandas as pd
import os
import argparse

def reformat_result(input_path:str, output_path:str, 
                    origin_path:str='/home/s3/hyeryung/mucoco/new_module/toxicity-avoidance/data/testset_gpt2_2500.jsonl', 
                    include_origin:bool=False,
                    nickname:str=None) -> None:
    """
    @param: input_path. path to decoding output file. e.g. "/home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/{display_name}/outputs_epsilon-3.txt"
    @param: output_path. path to save xlsx format of the output file. e.g., "/home/s3/hyeryung/mucoco/notebooks/evaluate/xlsx_outputs/{display_name}_outputs_epsilon-3.xlsx"
    @param: origin_path. path to the original gpt-2 output. 
    @param: include_origin. whether to interleave with original gpt-2 output.
    @param: nickname. nickname of the decoding setting.
    
    The function saves an excel file of format
    |index|nickname|prompt|text|losses|weighted_loss|allsat|
    |0-0|<nickname>|<prompt>|<original gpt-2 output>|'original'|'original'|'original'|
    |0-0|<nickname>|<prompt>|<updated output>|x|x|x|
    |0-1|<nickname>|<prompt>|<original gpt-2 output>|'original'|'original'|'original'|
    |0-1|<nickname>|<prompt>|<updated output>|x|x|x|
    |0-2|<nickname>|<prompt>|<original gpt-2 output>|'original'|'original'|'original'|
    |0-2|<nickname>|<prompt>|<updated output>|x|x|x|
    ...
    """
    
    dat = pd.read_json(input_path, lines=True)
    if nickname is None:
        nickname = '/'.join(input_path.split('/')[-3:-1])
    if include_origin:
        original_dat = pd.read_json(origin_path, lines=True)

    # dat.prompt = dat.prompt.applytext

    untangled_dat = {'index': [], 'nickname': [], 'prompt': [], 'text': [], 'allsat': [], 'losses': [], 'weighted_loss': []}
    for ix, row in dat.iterrows():
        
        prompt = row.prompt['text']
        if include_origin:
            corresponding_row = original_dat.loc[ix, :].copy()
            assert prompt == corresponding_row.prompt['text']
        for jx, gen in enumerate(row.generations):
            
            if ('losses' not in gen):
                untangled_dat['index'].append(f'{ix}-{jx}')
                untangled_dat['nickname'].append(nickname)
                untangled_dat['prompt'].append(prompt)
                untangled_dat['text'].append(gen['text'])
                untangled_dat['losses'].append('n/a')
                untangled_dat['weighted_loss'].append('n/a')
                untangled_dat['allsat'].append('n/a')
            elif ('losses' in gen and gen['text'] != ""):
                if include_origin:
                    # first append original text
                    untangled_dat['index'].append(f'{ix}-{jx}')
                    untangled_dat['nickname'].append('original')
                    untangled_dat['prompt'].append(prompt)
                    untangled_dat['text'].append(corresponding_row.generations[jx]['text'])
                    untangled_dat['losses'].append('original')
                    untangled_dat['weighted_loss'].append('original')
                    untangled_dat['allsat'].append('original')

                untangled_dat['index'].append(f'{ix}-{jx}')
                untangled_dat['nickname'].append(nickname)
                untangled_dat['prompt'].append(prompt)
                untangled_dat['text'].append(gen['text'])
                untangled_dat['losses'].append(gen['losses'])
                untangled_dat['weighted_loss'].append(gen['weighted_loss'])
                untangled_dat['allsat'].append(gen['allsat'])
        
    final_dat = pd.DataFrame(untangled_dat)
    # final_dat.to_excel(f"/home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/{display_name}/outputs_epsilon-3.xlsx")
    # os.makedirs(f"/home/s3/hyeryung/mucoco/notebooks/evaluate/xlsx_outputs/", exist_ok=True)
    final_dat.to_excel(output_path)        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--origin_path', default="")
    parser.add_argument('--include_origin', action='store_true')
    parser.add_argument('--nickname', default=None)
    args = parser.parse_args()
    
    if (args.origin_path != "") and (args.include_origin):
        reformat_result(args.input_path, args.output_path, args.origin_path, args.include_origin, nickname=args.nickname)
    else:
        reformat_result(args.input_path, args.output_path, nickname=args.nickname)