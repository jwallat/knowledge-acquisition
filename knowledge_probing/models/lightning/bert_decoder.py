from __future__ import annotations
from typing import List
from knowledge_probing.probing.metrics import calculate_metrics
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertForMaskedLM
import torch
import functools
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from knowledge_probing.models.decoder_head import MyDecoderHead
from knowledge_probing.datasets.text_dataset import TextDataset
from argparse import ArgumentParser, Namespace
from knowledge_probing.models.bert_model_util import saved_model_has_mlm_head, mask_tokens
from knowledge_probing.file_utils import find_checkpoint_in_dir
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
import sys


class BertDecoder(BaseDecoder):
    def __init__(self, hparams):
        super(BertDecoder, self).__init__()

        self.hparams = hparams
        self.model, self.config, self.decoder = self.prepare_model(
            hparams=self.hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_type)

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        self.total_num_training_steps = 0

        self.collate = functools.partial(
            self.mlm_collate, tokenizer=self.tokenizer)

    def prepare_model(self, hparams: Namespace):
        # Get config for Decoder
        config = AutoConfig.from_pretrained(hparams.model_type)
        config.output_hidden_states = True

        print('***********************************    LOADING MODEL    ****************************************')

        # Load Bert as BertModel which is plain and has no head on top
        if hparams.use_model_from_dir:
            print('Loading model from dir: ', hparams.model_dir)
            model = AutoModel.from_pretrained(hparams.model_dir, config=config)
        else:
            print('Loading model: ', hparams.model_type)
            model = AutoModel.from_pretrained(
                hparams.model_type, config=config)

        # Make sure the bert model is not trained
        model.eval()
        model.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False
        model.to(hparams.device)

        # Get the right decoder
        # Here we have several different option:
        # 1. Random initialization
        # 2. Pre-trained decoder. In this case we first check if the provided model has an own MLM-decoder associated.
        # Otherwise, we load a mlm-head from a pre-trained checkpoint
        decoder = None
        if hparams.decoder_initialization == 'random':
            print('Loading randomly initialized Decoder')
            decoder = MyDecoderHead(config)
        elif hparams.decoder_initialization == 'pre-trained':
            if saved_model_has_mlm_head(path=hparams.model_dir):
                print('Using models own mlm head from this model: ',
                      hparams.model_dir)
                mlm_head = (BertForMaskedLM.from_pretrained(
                    hparams.model_dir, config=config)).cls
                decoder = MyDecoderHead(
                    config=config, pre_trained_head=mlm_head)

            else:
                # Initialize with standard pre-trained mlm head
                print(
                    'Loading standard pre-trained head Decoder from this model: ', hparams.model_type)
                mlm_head = (BertForMaskedLM.from_pretrained(
                    hparams.model_type, config=config)).cls
                decoder = MyDecoderHead(
                    config=config, pre_trained_head=mlm_head)
        else:
            raise Exception(
                "Could not find a decoder/mlm-head for the model provided ({})".format(hparams.model_type))

        print('***************************************    END    **********************************************')

        return model, config, decoder

    def forward(self, inputs, masked_lm_labels, attention_mask=None, layer=None):

        # get attention mask
        if attention_mask == None:
            attention_mask = inputs.clone()
            attention_mask[attention_mask != self.tokenizer.pad_token_id] = 1
            attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0

        # get Berts embeddings
        bert_outputs = self.model(inputs, attention_mask=attention_mask)
        embeddings = bert_outputs[2][layer]

        # Feed embeddings into decoder
        decoder_outputs = self.decoder(
            embeddings, masked_lm_labels=masked_lm_labels)
        loss = decoder_outputs[0]
        prediction_scores = decoder_outputs[1]

        return loss, prediction_scores

    def probe(self, batch, layer: int, relation_args) -> List:
        input_ids_batch = batch['masked_sentences']
        attention_mask_batch = batch['attention_mask']

        input_ids_batch = input_ids_batch.to(self.hparams.device)
        attention_mask_batch = attention_mask_batch.to(self.hparams.device)

        # Get predictions from models
        outputs = self.forward(input_ids_batch, masked_lm_labels=input_ids_batch,
                               attention_mask=attention_mask_batch, layer=layer)

        batch_prediction_scores = outputs[1]

        metrics_elements_from_batch = []
        for i, prediction_scores in enumerate(batch_prediction_scores):
            prediction_scores = prediction_scores[None, :, :]
            metrics_element = calculate_metrics(
                batch, i, prediction_scores, precision_at_k=relation_args.precision_at_k, tokenizer=self.tokenizer)
            metrics_elements_from_batch.append(metrics_element)

        # For elements in batch, compute the metrics elements
        return metrics_elements_from_batch

    def set_to_train(self):
        """
        This function is intended to set everthing of the model to training mode (given hparams.unfreeze_transformer=True). 
        Otherwise, we only set the decoder to train.
        """
        self.decoder.train()
        self.decoder.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        print('Decoder set to train')

        if self.hparams.unfreeze_transformer:
            self.model.train()
            self.model.requires_grad = True
            for param in self.model.parameters():
                param.requires_grad = True
            print('T5 set to train')

    def set_to_eval(self):
        """
        This function is intended to set everthing of the model to evaluation mode. 
        """
        self.eval()
        self.requires_grad = False
        for param in self.parameters():
            param.requires_grad = False
        print('Everything set to evaluation')
