import os
import sys
import math
sys.path.append("/home/s3/hyeryung/mucoco")
os.chdir("/home/s3/hyeryung/mucoco")

from numpy import std
from tqdm import tqdm
import torch
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup

from notebooks.utils.load_ckpt import define_model

### hyperparameters
filtering =True # if True, filters training data with high standard deviation in labels

batch_size = 8
num_epochs = 10
min_lr = 5e-5
max_lr = 5e-5
weight_decay = 0.01
config = {'batch_size': batch_size, 'num_epochs': num_epochs, 'min_lr': min_lr, 'max_lr': max_lr, 'model': 'roberta-large'}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# wandb.init(project='formality', entity='hayleyson', config=config)

### data load 
data= pd.read_csv('data/formality/PT16/answers', delimiter='\t', names=['score', 'individual scores', 'na', 'text'])
data = data.sample(frac=1,random_state=999).reset_index(drop=True)#shuffle
train_size = math.ceil(len(data) * 0.9)

train_data = data.iloc[:train_size,:].copy()
valid_data = data.iloc[train_size:, :].copy()

if filtering: #only filter training data
    train_data['std'] = train_data['individual scores'].apply(lambda x: std([float(i) for i in str(x).split(',')]))
    train_data = train_data.loc[train_data['std'] < 1.5].copy() 
    del train_data['std']

del train_data['individual scores']
del train_data['na']
del valid_data['individual scores']
del valid_data['na']

## save train/valid data for reproducibility
if filtering:
    train_data.to_csv('data/formality/PT16/train_filtered.tsv', sep='\t', index=False)
else:
    train_data.to_csv('data/formality/PT16/train.tsv', sep='\t', index=False)
valid_data.to_csv('data/formality/PT16/valid.tsv', sep='\t', index=False)

max_save_num = 1
if filtering:
    ckpt_save_path = "models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds-filtered"
else:
    ckpt_save_path = "models/roberta-base-pt16-formality-regressor-with-gpt2-large-embeds"
os.makedirs(ckpt_save_path, exist_ok=True)

model, tokenizer = define_model(num_classes=1, device=device)

def collate_fn(batch):
    outputs = tokenizer([example['text'] for example in batch], padding=True, truncation=True, return_tensors="pt")
    outputs['labels'] = torch.Tensor([example['labels'] for example in batch])
    
    return outputs

train_dataset = Dataset.from_pandas(train_data)
train_dataset = train_dataset.rename_column('score', 'labels')
train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size,collate_fn=collate_fn)

print('num batches in an epoch:', len(train_loader))

valid_dataset = Dataset.from_pandas(valid_data)
valid_dataset = valid_dataset.rename_column('score', 'labels')
valid_loader = DataLoader(valid_dataset, shuffle=True,batch_size=batch_size,collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=weight_decay)

num_warmup_steps = len(train_loader)*num_epochs*0.1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_loader)*num_epochs)


step=0
best_val_loss = float("inf")
for epoch in tqdm(range(num_epochs)):
    
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, total=len(train_loader)):
        
        loss = model(input_ids = batch['input_ids'].to(device), 
                     labels = batch['labels'].to(device),
                     attention_mask = batch['attention_mask'].to(device)).loss
        loss.backward()
        train_loss += loss.item()
        wandb.log({'step':step, 'train_loss': loss.item(), 'learning_rate': scheduler.get_last_lr()[0]})
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step+=1

    
    valid_loss = 0
    for batch in valid_loader:
        model.eval()
        with torch.no_grad():
            loss = model(input_ids = batch['input_ids'].to(device), 
                            labels = batch['labels'].to(device),
                            attention_mask = batch['attention_mask'].to(device)).loss
            valid_loss += loss.item()
    wandb.log({'epoch':epoch, 'train_loss_per_epoch':train_loss/len(train_loader) , 'valid_loss_per_epoch': valid_loss/len(valid_loader), 
               'epoch_end_learning_rate': scheduler.get_last_lr()[0]})
    print('Epoch: ', epoch, 'Training Loss: ', train_loss/len(train_loader), 'Validation Loss: ', valid_loss/len(valid_loader))
    # break
    avg_val_loss = valid_loss/len(valid_loader)
    if avg_val_loss < best_val_loss:
        print("Saving checkpoint!")
        best_val_loss = avg_val_loss
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'val_loss': best_val_loss,
        #     },
        #     f"{ckpt_save_path}/epoch_{epoch}_best_ckpt.pt"
        # )
        tokenizer.save_pretrained(f"{ckpt_save_path}/epoch_{epoch}")
        model.save_pretrained(f"{ckpt_save_path}/epoch_{epoch}")
    
    # only save 1 checkpoints
        import os
        from operator import itemgetter
        fileData = {}
        test_output_dir = ckpt_save_path
        for fname in os.listdir(test_output_dir):
            if fname.startswith('epoch'):
                fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
            else:
                pass
        sortedFiles = sorted(fileData.items(), key=itemgetter(1))

        if len(sortedFiles) < max_save_num:
            pass
        else:
            delete = len(sortedFiles) - max_save_num
            for x in range(0, delete):
                one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                print (one_folder_name)
                os.system('rm -r ' + one_folder_name)
        print ('-----------------------------------')
    