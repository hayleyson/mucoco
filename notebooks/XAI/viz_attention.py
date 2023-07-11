import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def get_attention_per_token(cls_model, dataloader, layer_num=10, pad_token_id=None,
                            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                            save_plot=False, save_plot_dir="./notebooks/XAI/fig"):

    if save_plot:
        os.makedirs(save_plot_dir, exist_ok=True)
    cls_attns_soft_list = []
    input_id_list = []
    labels_list = []
    preds_list = []
    attn_mask_list = []
    cls_model.eval()
    cls_model.to(device)

    # first save model attention values, prediction, target value, input_ids, ...
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()} # move training data to gpu
        with torch.no_grad():
            outputs = cls_model.forward(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        ############### BEGIN main logic for looking at attentions for XAI ###############
        attentions = outputs["attentions"][layer_num] # output shape: (batch_size, num_heads, sequence_length, sequence_length)
        cls_attns = attentions.max(1)[0][:, 0] # Take attention of CLS token. output shape: (batch_size, sequence_length)
        cls_attns[batch["attention_mask"]==0.0] = -float("inf")
        cls_attns_soft = cls_attns.softmax(-1)
        ############### END main logic for looking at attentions for XAI ###############
        cls_attns_soft_list.append(cls_attns_soft)
        input_id_list.append(batch["input_ids"])
        labels_list.append(batch["labels"])
        preds_list.append(outputs["logits"].softmax(-1)[:,1])
        attn_mask_list.append(batch["attention_mask"])

    # then visualize the attention values in a heatmap-like plot.
    for batch_num in range(len(dataloader)):
        print("Batch: ", batch_num)
        for ix in range(len(cls_attns_soft_list[batch_num])):
            
            attn=cls_attns_soft_list[batch_num][ix][input_id_list[batch_num][ix]!= pad_token_id]
            attn=attn.cpu().unsqueeze(0).numpy()
            toks=tokenizer.convert_ids_to_tokens(input_id_list[batch_num][ix][input_id_list[batch_num][ix]!= pad_token_id])
            fig, ax = plt.subplots(figsize=(0.2 * len(toks), 0.5))
            # sns.heatmap(attn, cbar=False,vmin=0, vmax=1) # if you want a heatmap with 0 being black and 1 being the brightest color.
            sns.heatmap(attn, cbar=False, ax=ax) # if you want a heatmap with minimum value being black and maximum value being the brightest color.
            for _x in range(attn.shape[-1]):
                plt.text(_x, 0.8, toks[_x], size=8, color='gray', rotation=40)
            plot_title = f"{layer_num}th Layer Attn / Ground Truth: {labels_list[batch_num][ix]:.3f} / Pred: {preds_list[batch_num][ix]:.3f}"
            file_name = f"{save_plot_dir}/layer{layer_num}_gt{labels_list[batch_num][ix]:.3f}_pred{preds_list[batch_num][ix]:.3f}_ix{ix+32*batch_num}.png"
            plt.title(plot_title)
            if save_plot:
              plt.savefig(file_name, bbox_inches="tight")
            plt.show()
            
if __name__ == "__main__":
    
    ########### data load & preparation ###########
    # set device type
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Which cpu/cuda?: {DEVICE}")
    print(f"Which device?: {torch.cuda.current_device()}")
    
    # Define dataset
    data = load_dataset("sst2")
    # toxicity_data = load_dataset("csv", data_files="./notebooks/results/test_mucoco+add_preds2.csv")
    
    # load trained model & tokenizer
    checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
    labels=[0,1]
    config = AutoConfig.from_pretrained(checkpoint, num_labels=len(labels))
    config.output_attentions = True # set this option!
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        # return tokenizer(example["text"], truncation=True)
        return tokenizer(example["sentence"], truncation=True)

    tokenized_dataset=data.map(tokenize_function, batched=True)

    # remove unnecessary columns, rename columns
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence"]) # drop columns except input_id, labels, attention_mask
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    # select those samples that we want to debug (check attention scores of)
    tokenized_dataset_subset = tokenized_dataset["test"]

    # build data loader
    batch_size = 32
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    subset_dataloader = DataLoader(
        dataset=tokenized_dataset_subset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )
    
    ########### doing actual xai ###########
    get_attention_per_token(model, subset_dataloader, layer_num=4, pad_token_id=tokenizer.convert_tokens_to_ids('[PAD]'),
                            save_plot=True, save_plot_dir='fig2/')