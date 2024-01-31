
import abc
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as ptl
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import XLMRobertaTokenizerFast, XLMRobertaXLConfig, XLMRobertaXLModel


# from comet.encoders import str2encoder
from comet.modules import LayerwiseAttention

from comet.models.base import CometModel
from comet.encoders.base import Encoder
from comet.encoders.xlmr import XLMREncoder
from comet.models.lru_cache import tensor_lru_cache
from comet.models.pooling_utils import average_pooling, max_pooling
from comet.models.predict_pbar import PredictProgressBar
from comet.models.predict_writer import CustomWriter
from comet.models.utils import (
    OrderedSampler,
    Prediction,
    Target,
    flatten_metadata,
    restore_list_order,
)

       

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
from transformers.optimization import (Adafactor,
                                       get_constant_schedule_with_warmup)

from comet.models.base import CometModel
from comet.models.metrics import MCCMetric, RegressionMetrics
from comet.models.utils import LabelSet, Prediction, Target
from comet.modules import FeedForward

import os
from pathlib import Path
from typing import Union

import torch
import yaml
from huggingface_hub import snapshot_download

from comet.models.base import CometModel
from comet.models.multitask.unified_metric import UnifiedMetric
from comet.models.multitask.xcomet_metric import XCOMETMetric
from comet.models.ranking.ranking_metric import RankingMetric
from comet.models.regression.referenceless import ReferencelessRegression
from comet.models.regression.regression_metric import RegressionMetric
from comet.models.download_utils import download_model_legacy

class XLMRXLEncoder(XLMREncoder):
    """XLM-RoBERTA-XL Encoder encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
    """

    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model)
        if load_pretrained_weights:
            self.model = XLMRobertaXLModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = XLMRobertaXLModel(
                XLMRobertaXLConfig.from_pretrained(pretrained_model),
                add_pooling_layer=False,
            )
        self.model.encoder.output_hidden_states = True

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
        ) -> Dict[str, torch.Tensor]:
            last_hidden_states, _, all_layers,attentions = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=False,
            )
            return {
                "sentemb": last_hidden_states[:, 0, :],
                "wordemb": last_hidden_states,
                "all_layers": all_layers,
                "attention_mask": attention_mask,
                "attentions": attentions
            }

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str): Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face

        Returns:
            Encoder: XLMRXLEncoder object.
        """
        return XLMRXLEncoder(pretrained_model, load_pretrained_weights)
    



str2encoder2 = {
    "XLM-RoBERTa-XL": XLMRXLEncoder,
}

class CometModelCustom(CometModel):

  def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 1.0e-06,
        learning_rate: float = 1.5e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-large",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        layer_transformation: str = "softmax",
        layer_norm: bool = True,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        class_identifier: Optional[str] = None,
        load_pretrained_weights: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = str2encoder2[self.hparams.encoder_model].from_pretrained(
            self.hparams.pretrained_model, load_pretrained_weights
        )

        self.epoch_nr = 0
        if self.hparams.layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                layer_transformation=layer_transformation,
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=self.hparams.layer_norm,
            )
        else:
            self.layerwise_attention = None

        if self.hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        if self.hparams.keep_embeddings_frozen:
            self.encoder.freeze_embeddings()

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs
        self.mc_dropout = False  # Flag used to control usage of MC Dropout
        self.caching = False  # Flag used to control Embedding Caching

        # If not defined here, metrics will not live in the same device as our model.
        self.init_metrics()
  def predict(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 16,
        gpus: int = 1,
        devices: Union[List[int], str, int] = None,
        mc_dropout: int = 0,
        progress_bar: bool = True,
        accelerator: str = "auto",
        num_workers: int = None,
        length_batching: bool = True,
    ) -> Prediction:
        """Method that receives a list of samples (dictionaries with translations,
        sources and/or references) and returns segment-level scores, system level score
        and any other metadata outputed by COMET models. If `mc_dropout` is set, it
        also returns for each segment score, a confidence value.

        Args:
            samples (List[Dict[str, str]]): List with dictionaries with source,
                translations and/or references.
            batch_size (int): Batch size used during inference. Defaults to 16
            devices (Optional[List[int]]): A sequence of device indices to be used.
                Default: None.
            mc_dropout (int): Number of inference steps to run using MCD. Defaults to 0
            progress_bar (bool): Flag that turns on and off the predict progress bar.
                Defaults to True
            accelarator (str): Pytorch Lightning accelerator (e.g: 'cpu', 'cuda', 'hpu'
                , 'ipu', 'mps', 'tpu'). Defaults to 'auto'
            num_workers (int): Number of workers to use when loading and preparing
                data. Defaults to None
            length_batching (bool): If set to true, reduces padding by sorting samples
                by sequence length. Defaults to True.

        Return:
            Prediction object with `scores`, `system_score` and any metadata returned
                by the model.
        """
        if mc_dropout > 0:
            self.set_mc_dropout(mc_dropout)

        if gpus > 0 and devices is not None:
            assert len(devices) == gpus, AssertionError(
                "List of devices must be same size as `gpus` or None if `gpus=0`"
            )
        elif gpus > 0:
            devices = gpus
        else: # gpu = 0
            devices = "auto"

        sampler = SequentialSampler(samples)
        if length_batching and gpus < 2:
            try:
                sort_ids = np.argsort([len(sample["src"]) for sample in samples])
            except KeyError:
                sort_ids = np.argsort([len(sample["ref"]) for sample in samples])
            sampler = OrderedSampler(sort_ids)

        if num_workers is None:
            # Guideline for workers that typically works well.
            num_workers = 2 * gpus

        self.eval()
        dataloader = DataLoader(
            dataset=samples,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.prepare_for_inference,
            num_workers=num_workers,
        )
        if gpus > 1:
            pred_writer = CustomWriter()
            callbacks = [
                pred_writer,
            ]
        else:
            callbacks = []

        if progress_bar:
            enable_progress_bar = True
            callbacks.append(PredictProgressBar())
        else:
            enable_progress_bar = False

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer = ptl.Trainer(
            devices=devices,
            logger=False,
            callbacks=callbacks,
            accelerator=accelerator if gpus > 0 else "cpu",
            strategy="auto" if gpus < 2 else "ddp",
            enable_progress_bar=enable_progress_bar,
        )
        return_predictions = False if gpus > 1 else True
        predictions = trainer.predict(
            self, dataloaders=dataloader, return_predictions=return_predictions
        )

        if gpus > 1:
            torch.distributed.barrier()  # Waits for all processes to finish predict

        # If we are in the GLOBAL RANK we need to gather all predictions
        if gpus > 1 and trainer.is_global_zero:
            predictions = pred_writer.gather_all_predictions()
            # Delete Temp folder.
            pred_writer.cleanup()
            return predictions

        elif gpus > 1 and not trainer.is_global_zero:
            # If we are not in the GLOBAL RANK we will return None
            exit()

        scores = torch.cat([pred["scores"] for pred in predictions], dim=0).tolist()
        if "metadata" in predictions[0]:
            metadata = flatten_metadata([pred["metadata"] for pred in predictions])
        else:
            metadata = []
        if "attentions" in predictions[0]:
          attentions = [pred['attentions'] for pred in predictions]

        output = Prediction(scores=scores, system_score=sum(scores) / len(scores), attentions=attentions)

        # Restore order of samples!
        if length_batching and gpus < 2:
            output["scores"] = restore_list_order(scores, sort_ids)
            if metadata:
                output["metadata"] = Prediction(
                    **{k: restore_list_order(v, sort_ids) for k, v in metadata.items()}
                )
            return output
        else:
            # Add metadata to output
            if metadata:
                output["metadata"] = metadata

            return output
 


class UnifiedMetricCustom(CometModelCustom):
    """UnifiedMetric is a multitask metric that performs word-level classification along
    with sentence-level regression. This metric has the ability to work with and without
    reference translations.

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.9.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to True.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        warmup_steps (int): Warmup steps for LR scheduler.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 3.0e-06.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 3.0e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'microsoft/infoxlm-large'.
        sent_layer (Union[str, int]): Encoder layer to be used for regression task ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'sparsemax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'False'.
        word_layer (int): Encoder layer to be used for word-level classification. Defaults
            to 24.
        loss (str): Loss function to be used. Defaults to 'mse'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
        hidden_sizes (List[int]): Size of hidden layers used in the regression head.
            Defaults to [3072, 1024].
        activations (Optional[str]): Activation function used in the regression head.
            Defaults to 'Tanh'.
        final_activation (Optional[str]): Activation function used in the last layer of
            the regression head. Defaults to None.
        input_segments (Optional[List[str]]): List with input segment names to be used.
            Defaults to ["mt", "src", "ref"].
        word_level_training (bool): If True, the model is trained with multitask
            objective. Defaults to False.
        loss_lambda (float): Weight assigned to the word-level loss. Defaults to 0.65.
        error_labels (List[str]): List of severity labels for word-level training.
            Defaults to ['minor', 'major'].
        cross_entropy_weights (Optional[List[float]]):  Weights for each label in the
            error_labels + weight for the default 'O' label. Defaults to None.
        load_pretrained_weights (Bool): If set to False it avoids loading the weights
            of the pretrained model (e.g. XLM-R) before it loads the COMET checkpoint
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.9,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 3.0e-06,
        learning_rate: float = 3.0e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "microsoft/infoxlm-large",
        sent_layer: Union[str, int] = "mix",
        layer_transformation: str = "sparsemax",
        layer_norm: bool = True,
        word_layer: int = 24,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        input_segments: List[str] = ["mt", "src", "ref"],
        word_level_training: bool = False,
        loss_lambda: float = 0.65,
        error_labels: List[str] = ["minor", "major"],
        cross_entropy_weights: Optional[List[float]] = None,
        load_pretrained_weights: bool = True,
    ) -> None:
        super().__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            layer=sent_layer,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="unified_metric",
            load_pretrained_weights=load_pretrained_weights,
        )
        self.save_hyperparameters()
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        self.word_level = word_level_training
        if word_level_training:
            self.encoder.labelset = self.label_encoder
            self.hidden2tag = nn.Linear(self.encoder.output_units, self.num_classes)

        if len(self.hparams.input_segments) == 3:
            # By default 3rd input [mt:src:ref] has 50% weight,
            # 2nd input [mt:ref] 33% and 1st input [mt:src] has 16%
            self.input_weights_spans = torch.tensor([0.1667, 0.3333, 0.5])

        # This is None by default and we will use argmax during decoding yet, to control over
        # precision and recall we can set it to another value.
        self.decoding_threshold = None
        self.init_losses()

    def set_input_weights_spans(self, weights: torch.Tensor):
        """Used to set input weights in another.

        Args:
            weights (torch.Tensor): Tensor (size 3) with input weights."""
        assert weights.shape == (3,)
        self.input_weights_spans = weights

    def set_decoding_threshold(self, threshold: float = 0.5):
        """Used during decoding to control over precision and recall. It always assumes
        that the first label corresponds to "no-error" and the remaining labels
        correspond to different severities.

        When set to a value, the following rule is used to decide if a subword belong to
        an error: torch.sum(probs[1:]) > threshold.

        Args:
            threshold (float): Threshold to decide when"""
        self.decoding_threshold = threshold

    def init_metrics(self):
        """Initializes training and validation metrics"""
        # Train and Dev correlation metrics
        self.train_corr = RegressionMetrics(prefix="train")
        self.val_corr = nn.ModuleList(
            [RegressionMetrics(prefix=d) for d in self.hparams.validation_data]
        )
        if self.hparams.word_level_training:
            self.label_encoder = LabelSet(self.hparams.error_labels)
            self.num_classes = len(self.label_encoder.labels_to_id)
            # Train and Dev MCC
            self.train_mcc = MCCMetric(num_classes=self.num_classes, prefix="train")
            self.val_mcc = nn.ModuleList(
                [
                    MCCMetric(num_classes=self.num_classes, prefix=d)
                    for d in self.hparams.validation_data
                ]
            )

    def init_losses(self) -> None:
        """Initializes Loss functions to be used."""
        self.sentloss = nn.MSELoss()
        if self.word_level:
            if self.hparams.cross_entropy_weights:
                assert len(self.hparams.cross_entropy_weights) == self.num_classes
                loss_weights = torch.tensor(self.hparams.cross_entropy_weights)
            else:
                loss_weights = None

            self.wordloss = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=-1, weight=loss_weights
            )

    def requires_references(self) -> bool:
        """Unified models can be developed to exclusively use [mt, ref] or to use both
        [mt, src, ref]. Models developed to use the source will work in a quality
        estimation scenario but models trained with [mt, ref] won't!

        Return:
            [bool]: True if the model was trained to work exclusively with references.
        """
        if self.hparams.input_segments == ["mt", "ref"]:
            return True
        return False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to initialize a training Optimizer and learning
        rate scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
                List with Optimizers and a List with lr_schedulers.
        """
        params = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        params += [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.word_level:
            params += [
                {
                    "params": self.hidden2tag.parameters(),
                    "lr": self.hparams.learning_rate,
                },
            ]

        if self.layerwise_attention:
            params += [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        # If warmup setps are not defined we don't need a scheduler.
        if self.hparams.warmup_steps < 1:
            return [optimizer], []

        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
        )
        return [optimizer], [scheduler]

    def read_training_data(self, path: str) -> List[dict]:
        """Reads a csv file with training data.

        Args:
            path (str): Path to the csv file to be loaded.

        Returns:
            List[dict]: Returns a list of training examples.
        """
        df = pd.read_csv(path)
        # Deep copy input segments
        columns = self.hparams.input_segments[:]
        # Make sure everything except score is str type
        for col in columns:
            df[col] = df[col].astype(str)
        columns.append("score")
        df["score"] = df["score"].astype("float16")
        df = df[columns]
        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Reads a csv file with validation data.

        Args:
            path (str): Path to the csv file to be loaded.

        Returns:
            List[dict]: Returns a list of validation examples.
        """
        df = pd.read_csv(path)
        # Deep copy input segments
        columns = self.hparams.input_segments[:]
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
        # Make sure everything except score is str type
        for col in columns:
            df[col] = df[col].astype(str)
        columns.append("score")
        df["score"] = df["score"].astype("float16")
        df = df[columns]
        return df.to_dict("records")

    def concat_inputs(
        self,
        input_sequences: Tuple[Dict[str, torch.Tensor]],
        unified_input: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Prepares tokenized src, ref and mt for joint encoding by putting
        everything into a single contiguous sequence.

        Args:
            input_sequences (Tuple[Dict[str, torch.Tensor]]): Tokenized Source, MT and
                Reference.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Contiguous sequence.
        """
        model_inputs = OrderedDict()
        # If we are using source and reference we will have to create 3 different input
        if unified_input:
            mt_src, mt_ref = input_sequences[:2], [
                input_sequences[0],
                input_sequences[2],
            ]
            src_input, _, _ = self.encoder.concat_sequences(
                mt_src, return_label_ids=self.word_level
            )
            ref_input, _, _ = self.encoder.concat_sequences(
                mt_ref, return_label_ids=self.word_level
            )
            full_input, _, _ = self.encoder.concat_sequences(
                input_sequences, return_label_ids=self.word_level
            )
            model_inputs["inputs"] = (src_input, ref_input, full_input)
            model_inputs["mt_length"] = input_sequences[0]["attention_mask"].sum(dim=1)
            return model_inputs

        # Otherwise we will have one single input sequence that concatenates the MT
        # with SRC/REF.
        else:
            model_inputs["inputs"] = (
                self.encoder.concat_sequences(
                    input_sequences, return_label_ids=self.word_level
                )[0],
            )
            model_inputs["mt_length"] = input_sequences[0]["attention_mask"].sum(dim=1)
        return model_inputs

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "fit"
    ) -> Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Tokenizes input data and prepares targets for training.

        Args:
            sample (List[Dict[str, Union[str, float]]]): Mini-batch
            stage (str, optional): Model stage ('train' or 'predict'). Defaults to "fit".

        Returns:
            Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: Model input
                and targets.
        """
        inputs = {k: [d[k] for d in sample] for k in sample[0]}
        input_sequences = [
            self.encoder.prepare_sample(inputs["mt"], self.word_level, None),
        ]

        src_input, ref_input = False, False
        if ("src" in inputs) and ("src" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(inputs["src"]))
            src_input = True

        if ("ref" in inputs) and ("ref" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(inputs["ref"]))
            ref_input = True

        unified_input = src_input and ref_input
        model_inputs = self.concat_inputs(input_sequences, unified_input)
        if stage == "predict":
            return model_inputs["inputs"]

        scores = [float(s) for s in inputs["score"]]
        targets = Target(score=torch.tensor(scores, dtype=torch.float))

        if "system" in inputs:
            targets["system"] = inputs["system"]

        if self.word_level:
            # Labels will be the same accross all inputs because we are only
            # doing sequence tagging on the MT. We will only use the mask corresponding
            # to the MT segment.
            seq_len = model_inputs["mt_length"].max()
            targets["mt_length"] = model_inputs["mt_length"]
            targets["labels"] = model_inputs["inputs"][0]["label_ids"][:, :seq_len]

        return model_inputs["inputs"], targets

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            input_ids (torch.Tensor): Input sequence.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids for
                BERT-like models. Defaults to None.

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Dict[str, torch.Tensor]: Sentence scores and word-level logits (if
                word_level_training = True)
        """
        # print("Calling UnifiedMetricCustom.forward")
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )
        # print(f"Keys of output from self.encoder(input_ids,...) : {encoder_out.keys()}")

        # Word embeddings used for the word-level classification task
        if self.word_level:
            if (
                isinstance(self.hparams.word_layer, int)
                and 0 <= self.hparams.word_layer < self.encoder.num_layers
            ):
                wordemb = encoder_out["all_layers"][self.hparams.word_layer]
            else:
                raise Exception(
                    "Invalid model word layer {}.".format(self.hparams.word_layer)
                )

        # embeddings used for the sentence-level regression task
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )
        elif (
            isinstance(self.hparams.sent_layer, int)
            and 0 <= self.hparams.sent_layer < self.encoder.num_layers
        ):
            embeddings = encoder_out["all_layers"][self.hparams.sent_layer]
        else:
            raise Exception(
                "Invalid model sent layer {}.".format(self.hparams.word_layer)
            )
        sentemb = embeddings[:, 0, :] # We take the CLS token as sentence-embedding

        if self.word_level:
            sentence_output = self.estimator(sentemb)
            word_output = self.hidden2tag(wordemb)
            return Prediction(score=sentence_output.view(-1), logits=word_output, attentions=encoder_out['attentions'])

        return Prediction(score=self.estimator(sentemb).view(-1), attentions=encoder_out['attentions'])

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """Receives model batch prediction and respective targets and computes
        a loss value

        Args:
            prediction (Prediction): Batch prediction
            target (Target): Batch targets

        Returns:
            torch.Tensor: Loss value
        """
        sentence_loss = self.sentloss(prediction.score, target.score)
        if self.word_level:
            predictions = prediction.logits.reshape(-1, self.num_classes)
            targets = target.labels.reshape(-1).type(torch.LongTensor).cuda()
            word_loss = self.wordloss(predictions, targets)
            return sentence_loss * (1 - self.hparams.loss_lambda) + word_loss * (
                self.hparams.loss_lambda
            )
        else:
            return sentence_loss

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_nb: int
    ) -> torch.Tensor:
        """Pytorch Lightning training_step.

        Args:
            batch (Tuple[Dict[str, torch.Tensor]]): The output of your prepare_sample
                function.
            batch_nb (int): Integer displaying which batch this is.

        Returns:
            torch.Tensor: Loss value
        """
        batch_input, batch_target = batch
        # When using references our loss will be computed with 3 different forward
        # passes. Loss = L src + L ref + L src_and_ref
        predictions = [self.forward(**input_seq) for input_seq in batch_input]
        loss_value = 0
        for pred in predictions:
            if self.word_level:
                seq_len = batch_target.mt_length.max()
                pred.logits = pred.logits[:, :seq_len, :]
            loss_value += self.compute_loss(pred, batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.first_epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log(
            "train_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            batch_size=batch_target.score.shape[0],
            sync_dist=True,
        )
        return loss_value

    def validation_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_nb: int, dataloader_idx: int
    ) -> None:
        """Pytorch Lightning validation_step.

        Args:
            batch (Tuple[Dict[str, torch.Tensor]]): The output of your prepare_sample
                function.
            batch_nb (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this is.
        """
        batch_input, batch_target = batch
        predictions = [self.forward(**input_seq) for input_seq in batch_input]
        # Final score is the average of the 3 scores when using references.
        scores = torch.stack([pred.score for pred in predictions], dim=0).mean(dim=0)
        if self.word_level:
            seq_len = batch_target.mt_length.max()
            # Final probs for each word is the average of the 3 forward passes.
            subword_probs = [
                nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :]
                for o in predictions
            ]
            subword_probs = torch.mean(torch.stack(subword_probs), dim=0)
            # Removing masked targets and the corresponding logits.
            # This includes subwords and padded tokens.
            probs = subword_probs.reshape(-1, self.num_classes)
            targets = batch_target.labels.reshape(-1)
            mask = targets != -1
            probs, targets = probs[mask, :], targets[mask].int()

        if dataloader_idx == 0:
            self.train_corr.update(scores, batch_target.score)
            if self.word_level:
                self.train_mcc.update(probs, targets)

        elif dataloader_idx > 0:
            self.val_corr[dataloader_idx - 1].update(
                scores,
                batch_target.score,
                batch_target["system"] if "system" in batch_target else None,
            )
            if self.word_level:
                self.val_mcc[dataloader_idx - 1].update(probs, targets)

    # Overwriting this method to log correlation and classification metrics
    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics."""
        self.log_dict(self.train_corr.compute(), prog_bar=False, sync_dist=True)
        self.train_corr.reset()

        if self.word_level:
            self.log_dict(self.train_mcc.compute(), prog_bar=False, sync_dist=True)
            self.train_mcc.reset()

        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            corr_metrics = self.val_corr[i].compute()
            self.val_corr[i].reset()
            if self.word_level:
                cls_metric = self.val_mcc[i].compute()
                self.val_mcc[i].reset()
                results = {**corr_metrics, **cls_metric}
            else:
                results = corr_metrics

            # Log to tensorboard the results for this validation set.
            self.log_dict(results, prog_bar=False, sync_dist=True)
            val_metrics.append(results)

        average_results = {"val_" + k.split("_")[-1]: [] for k in val_metrics[0].keys()}
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + k.split("_")[-1]].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()},
            prog_bar=True,
            sync_dist=True,
        )

    def set_mc_dropout(self, value: int):
        """Sets Monte Carlo Dropout runs per sample.

        Args:
            value (int): number of runs per sample.
        """
        raise NotImplementedError("MCD not implemented for this model!")

    def decode(
        self,
        subword_probs: torch.Tensor,
        input_ids: torch.Tensor,
        mt_offsets: torch.Tensor,
    ) -> List[Dict]:
        """Decode error spans from subwords.

        Args:
            subword_probs (torch.Tensor): probabilities of each label for each subword.
            input_ids (torch.Tensor): input ids from the model.
            mt_offsets (torch.Tensor): subword offsets.

        Return:
            List with of dictionaries with text, start, end, severity and a
            confidence score which is the average of the probs for that label.
        """
        decoded_output = []
        for i in range(len(mt_offsets)):
            seq_len = len(mt_offsets[i])
            error_spans, in_span, span = [], False, {}
            for token_id, probs, token_offset in zip(
                input_ids[i, :seq_len], subword_probs[i][:seq_len], mt_offsets[i]
            ):
                if self.decoding_threshold:
                    if torch.sum(probs[1:]) > self.decoding_threshold:
                        probability, label_value = torch.topk(probs[1:], 1)
                        label_value += 1  # offset from removing label 0
                    else:
                        # This is just to ensure same format but at this point
                        # we will only look at label 0 and its prob
                        probability, label_value = torch.topk(probs[0], 1)
                else:
                    probability, label_value = torch.topk(probs, 1)

                # Some torch versions topk returns a shape 1 tensor with only
                # a item inside
                label_value = (
                    label_value.item()
                    if label_value.dim() < 1
                    else label_value[0].item()
                )
                label = self.label_encoder.ids_to_label.get(label_value)
                # Label set:
                # O I-minor I-major
                # Begin of annotation span
                if label.startswith("I") and not in_span:
                    in_span = True
                    span["tokens"] = [
                        token_id,
                    ]
                    span["severity"] = label.split("-")[1]
                    span["offset"] = list(token_offset)
                    span["confidence"] = [
                        probability,
                    ]

                # Inside an annotation span
                elif label.startswith("I") and in_span:
                    span["tokens"].append(token_id)
                    span["confidence"].append(probability)
                    # Update offset end
                    span["offset"][1] = token_offset[1]

                # annotation span finished.
                elif label == "O" and in_span:
                    error_spans.append(span)
                    in_span, span = False, {}

            sentence_output = []
            for span in error_spans:
                sentence_output.append(
                    {
                        "text": self.encoder.tokenizer.decode(span["tokens"]),
                        "confidence": torch.concat(span["confidence"]).mean().item(),
                        "severity": span["severity"],
                        "start": span["offset"][0],
                        "end": span["offset"][1],
                    }
                )
            decoded_output.append(sentence_output)
        return decoded_output

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """PyTorch Lightning predict_step

        Args:
            batch (Dict[str, torch.Tensor]): The output of your prepare_sample function
            batch_idx (Optional[int], optional): Integer displaying which batch this is
                Defaults to None.
            dataloader_idx (Optional[int], optional): Integer displaying which
                dataloader this is. Defaults to None.

        Returns:
            Prediction: Model Prediction
        """
        # print("Calling UnifiedMetric.predict_step")
        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Final score is the average of the 3 scores!
            avg_scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            batch_prediction = Prediction(
                scores=avg_scores,
                metadata=Prediction(
                    src_scores=predictions[0].score,
                    ref_scores=predictions[1].score,
                    unified_scores=predictions[2].score,
                ),
            )
            if self.word_level:
                mt_mask = batch[0]["label_ids"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = [
                    nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :] * w
                    for w, o in zip(self.input_weights_spans, predictions)
                ]
                subword_probs = torch.sum(torch.stack(subword_probs), dim=0)
                error_spans = self.decode(
                    subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
                )
                batch_prediction.metadata["error_spans"] = error_spans

        else:
            # print(f"len(batch): {len(batch)}")
            model_output = self.forward(**batch[0])
            # print(f"len of model_output.attentions: {len(model_output.attentions)}")
            # print(f"shape of model_output.attentions[0]: {model_output.attentions[0].shape}")
            batch_prediction = Prediction(scores=model_output.score, attentions=model_output.attentions)
            if self.word_level:
                mt_mask = batch[0]["label_ids"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = nn.functional.softmax(model_output.logits, dim=2)[
                    :, :seq_len, :
                ]
                error_spans = self.decode(
                    subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
                )
                batch_prediction = Prediction(
                    scores=model_output.score,
                    metadata=Prediction(error_spans=error_spans),
                    attentions=model_output.attentions
                )
        return batch_prediction
    



str2model = {
    "referenceless_regression_metric": ReferencelessRegression,
    "regression_metric": RegressionMetric,
    "ranking_metric": RankingMetric,
    "unified_metric": UnifiedMetricCustom,
    "xcomet_metric": XCOMETMetric,
}


def download_model(
    model: str,
    saving_directory: Union[str, Path, None] = None,
    local_files_only: bool = False,
) -> str:
    try:
        model_path = snapshot_download(
            repo_id=model, cache_dir=saving_directory, local_files_only=local_files_only
        )
    except Exception:
        try:
            checkpoint_path = download_model_legacy(model, saving_directory)
        except Exception:
            raise KeyError(f"Model '{model}' not supported by COMET.")
    else:
        checkpoint_path = os.path.join(*[model_path, "checkpoints", "model.ckpt"])
    return checkpoint_path


def load_from_checkpoint(
    checkpoint_path: str, reload_hparams: bool = False, strict: bool = False
) -> CometModel:
    """Loads models from a checkpoint path.

    Args:
        checkpoint_path (str): Path to a model checkpoint.
        reload_hparams (bool): hparams.yaml file located in the parent folder is
            only use for deciding the `class_identifier`. By setting this flag
            to True all hparams will be reloaded.
        strict (bool): Strictly enforce that the keys in checkpoint_path match the
            keys returned by this module's state dict. Defaults to False
    Return:
        COMET model.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise Exception(f"Invalid checkpoint path: {checkpoint_path}")

    parent_folder = checkpoint_path.parents[1]  # .parent.parent
    hparams_file = parent_folder / "hparams.yaml"

    if hparams_file.is_file():
        with open(hparams_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model_class = str2model[hparams["class_identifier"]]
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            load_pretrained_weights=False,
            hparams_file=hparams_file if reload_hparams else None,
            map_location=torch.device("cpu"),
            strict=strict,
        )
        return model
    else:
        raise Exception(f"hparams.yaml file is missing from {parent_folder}!")