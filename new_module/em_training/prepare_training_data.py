
    ### data load
    # data= pd.read_csv('data/formality/PT16/answers', delimiter='\t', names=['score', 'individual scores', 'na', 'text'])
    # data = data.sample(frac=1,random_state=999).reset_index(drop=True)#shuffle
    # train_size = math.ceil(len(data) * 0.9)

    # train_data = data.iloc[:train_size,:].copy()
    # valid_data = data.iloc[train_size:, :].copy()

    # if filtering: #only filter training data
    #     train_data['std'] = train_data['individual scores'].apply(lambda x: std([float(i) for i in str(x).split(',')]))
    #     train_data = train_data.loc[train_data['std'] < 1.5].copy()
    #     del train_data['std']

    # del train_data['individual scores']
    # del train_data['na']
    # del valid_data['individual scores']
    # del valid_data['na']

    # ## save train/valid data for reproducibility
    # if filtering:
    #     train_data.to_csv('data/formality/PT16/train_filtered.tsv', sep='\t', index=False)
    # else:
    #     train_data.to_csv('data/formality/PT16/train.tsv', sep='\t', index=False)
    # valid_data.to_csv('data/formality/PT16/valid.tsv', sep='\t', index=False)
