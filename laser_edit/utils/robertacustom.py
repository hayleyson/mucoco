"""
Custom Roberta models whose embedding layer comprises a linear layer 
that maps frozen embedding weights from another model 
to the learned embeddings of the RoBERTa model.

The code is to run MuCoLa as a baseline and adapted from MuCoLa's git repository (https://github.com/Sachin19/mucoco/tree/sampling2).
"""

import logging, os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaPreTrainedModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
)


logger = logging.getLogger(__name__)

class RobertaCustomForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        print(config.vocab_size)


        self.roberta = RobertaModel(config, add_pooling_layer=False)
        embeds = self.roberta.get_input_embeddings()
        old_dim = getattr(config,'n_embd', embeds.embedding_dim)
        new_dim = getattr(config,'new_n_embd', None)
        new_vocab_size = getattr(config,'new_vocab_size', config.vocab_size)
        if new_dim is not None:
            new_embeds = nn.Sequential(nn.Embedding(new_vocab_size, new_dim), nn.Linear(new_dim, old_dim, bias=False))
            self.roberta.set_input_embeddings(new_embeds)

        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def define_model(num_classes:int = 2,
                 mod_path:str=None, 
                 load_weights:bool=True, 
                 output_attentions:bool=False, 
                 output_hidden_states:bool=False,
                 device:torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 embedding_model:str='gpt2-large',
                 encoder_model:str='roberta-base',
                 task:str=None)-> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:

    tokenizer_ = AutoTokenizer.from_pretrained(encoder_model)
    if embedding_model != "none":
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)
    else:
        tokenizer = tokenizer_
        # tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        
    config = AutoConfig.from_pretrained(encoder_model, num_labels=num_classes)
    config2 = None
    if embedding_model != "none":  
        config2 = AutoConfig.from_pretrained(embedding_model, num_labels=num_classes)
        # print(config2.pad_token_id)
        config2.pad_token_id = tokenizer.pad_token_id
        # print(config2.pad_token_id)
        # print("look above for padding")

        tokenizer_ = AutoTokenizer.from_pretrained(encoder_model, config=config)
        tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)

    SPECIAL_TOKENS = {}
    if "pad_token" not in tokenizer.special_tokens_map:
        SPECIAL_TOKENS.update({"pad_token": tokenizer.eos_token})
    if ("bos_token" not in tokenizer.special_tokens_map) and (task == "nli"):
        SPECIAL_TOKENS.update({"bos_token": tokenizer.eos_token})
    if ("sep_token" not in tokenizer.special_tokens_map) and (task == "nli"):
        SPECIAL_TOKENS.update({"sep_token": tokenizer.eos_token})
    # config.pad_token_id = tokenizer.eos_token_id
    # print("Adding special tokens")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print(tokenizer.special_tokens_map)

    # if embedding_model != "none":
    model = AutoModelForSequenceClassification.from_pretrained(embedding_model, config=config2) # unindented
    # model.resize_token_embeddings(len(tokenizer))

    def learn_vecmap(X, y):
        # print("computing vecmap")
        w = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(y)
        vecmap = torch.nn.Linear(w.size(0), w.size(1), bias=False)
        # print(w.size(), vecmap.weight.size())
        vecmap.weight.data.copy_(w.data.t())
        return vecmap

    def vocab_permutation(vocab1, vocab2):
        vocab2itos = {k:v for v,k in vocab2.items()}
        vocab2list = [vocab2itos[k] for k in range(len(vocab2itos))]

        perm1 = []
        perm2 = []
        unincluded = []
        for i, word in enumerate(vocab2list):
            if word in vocab1:
                perm1.append(vocab1[word])
                perm2.append(i)
            else:
                unincluded.append(word)

        # print(unincluded)
        return perm1, perm2

    embeds = model.get_input_embeddings()
    new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
    for p in new_embeds.parameters():
        p.requires_grad = False

    new_embeds.weight.data.copy_(embeds.weight)
    config.new_n_embd = new_embeds.embedding_dim
    config.new_vocab_size = new_embeds.num_embeddings
    config.output_attentions=output_attentions
    config.output_hidden_states=output_hidden_states

    model_ = AutoModelForSequenceClassification.from_pretrained(encoder_model, config=config)
    tokenizer_ = AutoTokenizer.from_pretrained(encoder_model, config=config)

    perm, perm_ = vocab_permutation(tokenizer.vocab, tokenizer_.vocab)
    old_embeds = model_.get_input_embeddings()
    vecmap = learn_vecmap(new_embeds.weight[perm], old_embeds.weight[perm_])
    new_embeds = torch.nn.Sequential(new_embeds, vecmap)
    model_.set_input_embeddings(new_embeds)
    model = model_

    print("DEVICE: ", device)

    # state_dict
    if load_weights and (mod_path is not None):
        mod = torch.load(mod_path, map_location=device)
        model.load_state_dict(mod)

    model.to(device)

    return model, tokenizer
    