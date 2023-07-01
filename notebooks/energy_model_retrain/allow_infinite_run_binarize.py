# -*- coding: utf-8 -*-
from torch import nn
import torch
import scipy
# from scipy.special import softmax

from datetime import datetime
import wandb
import torch
import os
import json
###############################################################################################################
import click
from glob import glob 
import yaml
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AddedToken
from torch import optim

import numpy as np
import pandas as pd

from evaluate import load 
import logging
###############################################################################################################

import os
# os.chdir('/home/hyeryungson/mucoco/notebooks/energy-model-retrain')
from mucoco.notebooks.utils.load_ckpt import define_model
from notebooks.energy_model_retrain.customTrainer import CustomTrainer
# os.chdir('/home/hyeryungson/mucoco')
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "DEBUG"))
formatter = '%(asctime)s:%(module)s:%(levelname)s:%(message)s'
logging.basicConfig(format=formatter)

@click.command()
@click.option('--resume', default=False, help='Whether to resume previously stopped run')
@click.option('--epochs', default=10, help='Total number of training epochs')
@click.option('--warmup_steps', default=600, help='Number of steps for learning rate warm up')
@click.option('--learning_rate', default=1e-5, help='Initial learning rate')
@click.option('--weight_decay', default=0.01, help='Strength of weight decay for AdamW')
@click.option('--logging_steps', default=100, help='Number of steps for which to log')
@click.option('--eval_steps', default=500, help='Number of steps for which to evaluate')
@click.option('--evaluation_strategy', default="steps", help='Evaluation strategy')
@click.option('--save_strategy', default="steps", help='Save strategy')
@click.option('--save_total_limit', default=5, help='Number of checkpoints to keep track of.')
@click.option('--per_device_train_batch_size', default=4, help='Batch size per device during training')
@click.option('--per_device_eval_batch_size', default=4, help='Batch size for evaluation')
@click.option('--gradient_accumulation_steps', default=4, help='Steps to accumulate gradients')
@click.option('--metric_for_best_model', default="eval_loss", help='Metric to decide best model after training completes')
@click.option('--greater_is_better', default=False, help='Whether metric_for_best_model is the better the greater')
@click.option('--wandb_name', default='energy-model', help='Display name of the run for wandb')
@click.option('--run_id', default=None, help='Wandb run id to resume')
def main(resume, epochs, warmup_steps, learning_rate, weight_decay, logging_steps, eval_steps, evaluation_strategy, save_strategy, save_total_limit, 
         per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, metric_for_best_model, greater_is_better, wandb_name, run_id):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits = eval_pred.predictions
        # labels = eval_pred.label_ids
        predictions = scipy.special.softmax(logits, axis=-1)
        # log_predictions = np.log(predictions)
        # all_losses = np.sum(labels * log_predictions, axis=-1)
        # cross_entropy = -np.mean(all_losses, axis=0)
        
        # mse
        mse_metric = load("mse")
        
        # mae
        mae_metric = load("mae")
        
        return {"mse": mse_metric.compute(predictions=predictions[:, 1], references=labels[:, 1]),
                "mae": mae_metric.compute(predictions=predictions[:, 1], references=labels[:, 1])}

    # define arguments
    params = ['', 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification',
    '0,1',
    'train',
    'dev',
    'test',
    'roberta-base',
    'models_binarize/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds',
    'gpt2-roberta',
    'full',
    'gpt2-large',
    'freeze-vecmap',
    'dontbinarize',
    'jsonl']

    time_limit = 1
    # resume_yn=True
    resume_yn=resume
    logger.debug("Is CUDA Available: %s", torch.cuda.is_available())
    logger.debug("GPU count: %s", torch.cuda.device_count())
    os.makedirs(params[7], exist_ok=True)

    if resume_yn:
        # # get the latest run_id for now
        # run_id=sorted(glob(f'wandb/run-*'), reverse=True)[0].split('-')[-1]
        logger.info(f'Resuming run_id: {run_id}')
        
        # for resume in wandb
        wandb.init(project="huggingface", resume="must", id=run_id, name=wandb_name)
        wandb_path=sorted(glob(f'wandb/run-*{run_id}'), reverse=True)[0]
        config_path=os.path.join(wandb_path, 'files/config.yaml')
        # logger.debug('path to yaml file %s', config_path)
        with open(config_path, 'r') as stream:
            past_run_config = yaml.safe_load(stream)
            
        # set the ckpt with highest step number as ckpt path
        ckpt_dir=sorted(glob(f'{params[7]}/results/*/'), key=lambda x: int(x.split('-')[-1].strip('/')), reverse=True)[0]
        logger.info('Resuming from... %s', ckpt_dir)
        # load model from ckpt
        model, config, tokenizer = define_model(mod_path=os.path.join(ckpt_dir, "pytorch_model.bin"), load_weights=True)
    else:
        wandb.init(project="huggingface", name=wandb_name)
        model, config, tokenizer = define_model(mod_path=None, load_weights=False)
        tokenizer.save_pretrained(f"{params[7]}/checkpoint_best")
        
    train_data = pd.read_json('/home/hyeryungson/mucoco/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/train.jsonl', lines=True)
    dev_data = pd.read_json('/home/hyeryungson/mucoco/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/dev.jsonl', lines=True)
    test_data = pd.read_json('/home/hyeryungson/mucoco/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/fine-grained/test.jsonl', lines=True)

    # logger.debug('train_data shape %s', train_data.shape)
    # logger.debug('dev_data shape %s', dev_data.shape)
    # logger.debug('test_data shape %s', test_data.shape)
    
    train_texts, train_labels = train_data['text'].tolist(), train_data['toxicity'].tolist()
    val_texts, val_labels = dev_data['text'].tolist(), dev_data['toxicity'].tolist()
    test_texts, test_labels = test_data['text'].tolist(), test_data['toxicity'].tolist()

    # binarize labels
    cutoff = 0.5
    train_labels = [1.0 if x >= cutoff else 0.0 for x in train_labels]
    val_labels = [1.0 if x >= cutoff else 0.0 for x in val_labels]
    test_labels = [1.0 if x >= cutoff else 0.0 for x in test_labels]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor([1 - self.labels[idx], self.labels[idx]])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    logger.debug("epochs: %s", epochs)
    training_args = TrainingArguments(
        output_dir=f'{params[7]}/results',          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,   # batch size for evaluation
        warmup_steps=warmup_steps, # commented out for resume
        weight_decay=weight_decay,               # strength of weight decay # commented out for resume
        learning_rate=learning_rate, # commented out for resume
        logging_dir=f'{params[7]}/logs',            # directory for storing logs
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        eval_steps=eval_steps,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        gradient_accumulation_steps=gradient_accumulation_steps,
        load_best_model_at_end=True,
        report_to="wandb"
    )

    # logger.debug(training_args.n_gpu)

    trainer = CustomTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, lr_sch)
    )

    if resume_yn:
        # resume_from_checkpoint = sorted(glob(os.path.join(params[7] + '/results/*')), key=lambda x: int(x.split('-')[-1]), reverse=True)
        resume_from_checkpoint = ckpt_dir
    else:
        resume_from_checkpoint = None 
    logger.debug('resume_from_checkpoint %s', resume_from_checkpoint)


    try: 
        logger.debug("start training")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint, time_limit=time_limit)
        logger.debug("training finished")


        os.makedirs(f"{params[7]}/checkpoint_best", exist_ok=True)

        trainer.save_model(output_dir=f"{params[7]}/checkpoint_best") 
        logger.debug("model saved")

        logger.debug("running evaluation now")

        metrics = trainer.evaluate(val_dataset)
        logger.debug("validation %s", metrics)
        metrics = trainer.evaluate(test_dataset)
        logger.debug("test %s", metrics)

    except Exception as e:
        logger.critical(e)
    #     # if e.args[0] == "TIMEOUT":
    #     #     logger.critical(e.args[1])
    
if __name__ == "__main__":
    
    main()