import pandas as pd
from glob import glob
import random
import os

cur_dir = os.path.dirname(__file__)

def merge_xlsx_results():
    fpaths = glob(os.path.join(cur_dir, 'xlsx_outputs', '*.xlsx'))
    df_list = []
    
    for fpath in fpaths:
        
        cur_df = pd.read_excel(fpath, index_col=0)
        
        df_list.append(cur_df)
    
    fin_df = pd.concat(df_list, axis=0)
    fin_df['primary_loss'] = fin_df['losses'].apply(lambda x:eval(x)[0] if type(x) != float else x)
    fin_df['toxicity_loss'] = fin_df['losses'].apply(lambda x:eval(x)[1] if type(x) != float else x)
    
    fin_df = fin_df.sort_values(by=['index', 'nickname'], ascending=[True, True])
    fin_df.to_excel(os.path.join(cur_dir, 'result_merged.xlsx'))
    
def get_qual_eval_data():
    
    # my_method_data = pd.read_excel(os.path.join(cur_dir,'xlsx_outputs/locate-unit-gpt2-word-netps3-nls4-nps1-modelenergy_result.xlsx'),index_col=0)
    # my_method_data = pd.read_excel(os.path.join(cur_dir, 'xlsx_outputs/mlm-beamsearch-v1-token-nps5-k10-beam3-allsat_primary-closs0.43-t7w3q9xu_result.xlsx'), index_col=0)
    # my_method_data = pd.read_excel(os.path.join(cur_dir,'xlsx_outputs/sweep-mlm-beamsearch-v1.1-token-nps5-k10-beam3-weighted_sum-f67kbovx-wandb-wandb_result.xlsx'),index_col=0)
    my_method_data = pd.read_excel(os.path.join(cur_dir,'xlsx_outputs/sweep-mlm-beamsearch-v0.1-token-nps5-k10-beam3-weighted_sum-k9ot5vd7-wandb_result.xlsx'),index_col=0)
    
    mucola_data = pd.read_excel(os.path.join(cur_dir, 'xlsx_outputs/mucola_result.xlsx'), index_col=0)
    gpt2_data = pd.read_excel(os.path.join(cur_dir, 'xlsx_outputs/gpt2_result.xlsx'), index_col=0)
    
    ## get indices where original gpt2 output has more than 1 word
    filtered_gpt2_gen_ids = gpt2_data.loc[gpt2_data['text'].apply(lambda x: len(x.split(' ')))> 1,'index'].tolist()
    
    ## get indices where my method and mucola both have generations
    my_method_ids = my_method_data['index'].tolist()
    mucola_ids = mucola_data['index'].tolist()
    
    common_ids = set(my_method_ids).intersection(set(mucola_ids)).intersection(set(filtered_gpt2_gen_ids))
    print(f"total {len(common_ids)} common indexes")
    
    ## sample 30 of them
    random.seed(999)
    sample_common_ids = random.sample(common_ids, 30)
    
    fin_df = gpt2_data.loc[gpt2_data['index'].isin(sample_common_ids), :]
    fin_df = pd.concat([fin_df, my_method_data.loc[my_method_data['index'].isin(sample_common_ids), :]])
    fin_df = pd.concat([fin_df, mucola_data.loc[mucola_data['index'].isin(sample_common_ids),:]])

    fin_df = fin_df.sort_values(by=['index', 'nickname'], ascending=[True, True])
    fin_df = fin_df.reset_index(drop=True)
    fin_df.to_excel(os.path.join(cur_dir, 'qual_eval_data_beamsearch-v0.1.xlsx'))
    

if __name__ == "__main__":
    # merge_xlsx_results()
    get_qual_eval_data()
