from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaLMHead
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
# from accelerate.hooks import add_hook_to_module

import torch.nn as nn
import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

import logging
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)


class RobertaCustomPreTrainedModel(PreTrainedModel):

    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RobertaEmbeddings", "RobertaSelfAttention"]
        
    def add_mask_token_embedding(
        self, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Sequential:
        
        model_embeds = self._add_mask_token_embedding(pad_to_multiple_of)

        # Update base model and current model config
        self.config.vocab_size = model_embeds[0].weight.shape[0]
        self.vocab_size = model_embeds[0].weight.shape[0]

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _add_mask_token_embedding(self, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_num_tokens = old_embeddings[0].weight.shape[0] + 1
        new_embeddings = self._get_embeddings_with_mask_token_embedding(old_embeddings, new_num_tokens, pad_to_multiple_of)
        
        self.set_input_embeddings(new_embeddings)

        new_num_tokens = new_embeddings[0].weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_embeddings_with_mask_token_embedding(
        self,
        old_embeddings: nn.Sequential,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Sequential:

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        # Sequential(
        #   (0): Embedding(50257, 1280)
        #   (1): Linear(in_features=1280, out_features=768, bias=False)
        # )
        old_num_tokens, old_intermediate_embedding_dim = old_embeddings[0].weight.size()
        
        # Build new embeddings
        new_intermediate_embeddings = nn.Embedding(
            new_num_tokens,
            old_intermediate_embedding_dim,
            device=old_embeddings[0].weight.device,
            dtype=old_embeddings[0].weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_intermediate_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        new_intermediate_embeddings.weight.data[:n, :] = old_embeddings[0].weight.data[:n, :]
        
        # also set the weight of the last token by taking inverse. y W^T (W W^T)^{-1} 
        ## -> for numerical stability, instead of taking inverse, use linalg.solve
        ## 1) get the mask token embedding of Roberta
        mlm = AutoModelForMaskedLM.from_pretrained('roberta-base')
        mlm_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        y = mlm.get_input_embeddings().weight.data[mlm_tokenizer.mask_token_id, :].unsqueeze(-1)
        
        ## 2) calculate this -> we want to find solution x that satisfies W x = y where W is the embedding projection layer in the custom roberta that projects gpt2 embedding into roberta embedding
        ## What we hope is to find x such that it is projected to a mask token embedding and EM converted to MLM can hopefully recognize the embedding and treat it properly by generating candidates to fill it in. 
        ## Since W is not a square matrix, first rewrite the equation as WTW x = WTy -> Now, it can be seen as Ax = B where A = WTW, B = WTy
        W = old_embeddings[1].weight
        A = W.t().matmul(W)
        B = W.t().matmul(y)
        x = torch.linalg.solve(A, B)
        
        ## 3) set the weight of the last token
        new_intermediate_embeddings.weight.data[-1, :] = x.data.squeeze()

        # update requires_grad
        old_embeddings_requires_grad = old_embeddings[0].weight.requires_grad
        new_intermediate_embeddings.requires_grad_(old_embeddings_requires_grad)

        new_embeddings = nn.Sequential(new_intermediate_embeddings, 
                                       old_embeddings[1])
            
        return new_embeddings
    

    

class RobertaCustomForSequenceClassification(RobertaCustomPreTrainedModel):
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


class RobertaCustomForMaskedLM(RobertaCustomPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )


        self.num_labels = config.num_labels
        self.config = config
        # print(config.vocab_size)


        self.roberta = RobertaModel(config, add_pooling_layer=False)
        embeds = self.roberta.get_input_embeddings()
        old_dim = getattr(config,'n_embd', embeds.embedding_dim)
        new_dim = getattr(config,'new_n_embd', None)
        new_vocab_size = getattr(config,'new_vocab_size', config.vocab_size)
        if new_dim is not None:
            new_embeds = nn.Sequential(nn.Embedding(new_vocab_size, new_dim), nn.Linear(new_dim, old_dim, bias=False))
            self.roberta.set_input_embeddings(new_embeds)

        self.lm_head = RobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)

            masked_lm_loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(prediction_scores.device)
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


            
# @add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
# class RobertaForMaskedLM(RobertaPreTrainedModel):
#     _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

#     def __init__(self, config):
#         super().__init__(config)

#         if config.is_decoder:
#             logger.warning(
#                 "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
#                 "bi-directional self-attention."
#             )

#         self.roberta = RobertaModel(config, add_pooling_layer=False)
#         self.lm_head = RobertaLMHead(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_output_embeddings(self):
#         return self.lm_head.decoder

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head.decoder = new_embeddings

#     @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=MaskedLMOutput,
#         config_class=_CONFIG_FOR_DOC,
#         mask="<mask>",
#         expected_output="' Paris'",
#         expected_loss=0.1,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
#             config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
#             loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
#         kwargs (`Dict[str, any]`, optional, defaults to *{}*):
#             Used to hide legacy arguments that have been deprecated.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]
#         prediction_scores = self.lm_head(sequence_output)

#         masked_lm_loss = None
#         if labels is not None:
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(prediction_scores.device)
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

#         return MaskedLMOutput(
#             loss=masked_lm_loss,
#             logits=prediction_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# class RobertaLMHead(nn.Module):
#     """Roberta Head for masked language modeling."""

#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))
#         self.decoder.bias = self.bias

#     def forward(self, features, **kwargs):
#         x = self.dense(features)
#         x = gelu(x)
#         x = self.layer_norm(x)

#         # project back to size of vocabulary with bias
#         x = self.decoder(x)

#         return x