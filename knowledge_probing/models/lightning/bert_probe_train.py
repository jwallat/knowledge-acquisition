from __future__ import annotations
from typing import List

from pytorch_lightning.core.memory import parse_batch_shape
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
from knowledge_probing.datasets.cloze_dataset_train import TrainClozeDataset
from torch.nn.utils.rnn import pad_sequence
import sys
import random


class BertProbeTrain(BaseDecoder):
    def __init__(self, hparams):
        super(BertProbeTrain, self).__init__()

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

        self.probe_cloze_collate = functools.partial(
            self.train_cloze_collate, tokenizer=self.tokenizer)

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

    def prepare_data(self):
        self.train_dataset = TrainClozeDataset(
            probing_model=self, tokenizer=self.tokenizer, args=self.hparams)
        self.eval_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.valid_file, block_size=self.tokenizer.model_max_length)
        self.test_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.test_file, block_size=self.tokenizer.model_max_length)

    def train_cloze_collate(self, examples, tokenizer: AutoTokenizer):
        """ 
            This is a function that makes sure all entries in the batch are padded 
            to the correct length.
        """
        # print('Collate:')
        # print('examples: ', examples)
        sequences = [x['sequence'] for x in examples]
        obj_labels = [x['obj_label'] for x in examples]

        padded_sequences = pad_sequence(
            sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        # attention_mask = padded_sequences.clone()
        # attention_mask[attention_mask != tokenizer.pad_token_id] = 1
        # attention_mask[attention_mask == tokenizer.pad_token_id] = 0

        examples_batch = {
            "sequence": padded_sequences,
            "obj_label": obj_labels
            # "attention_mask": attention_mask,
        }

        return examples_batch

    def probe_mask_tokens(self, batch, tokenizer, hparams):
        all_labels = []
        all_input_ids = []

        # print('Mask tokens:')
        # print('batch: ', batch)

        for i, sequence in enumerate(batch['sequence']):
            # print('Item in batch: ', sequence)
            input_ids = sequence
            labels = sequence.clone()

            obj_label = batch['obj_label'][i]
            # print('label: ', obj_label)
            obj_label_id = tokenizer.convert_tokens_to_ids(obj_label)

            if 'object' in hparams.capacity_masking_mode:
                # print('Sentence: ', self.tokenizer.decode(input_ids))
                # print('Input ids: ', input_ids.tolist())
                # print('Obj: ', obj_label)
                # print('Looking for the index: ', obj_label_id)
                try:
                    obj_index = input_ids.tolist().index(obj_label_id)
                    input_ids[obj_index] = tokenizer.mask_token_id
                except:
                    print('****************** obj label not in input ids!!!!!!!!!!!!!')
            elif 'random' in hparams.capacity_masking_mode:
                random_index = random.randint(0, len(input_ids) - 1)
                input_ids[random_index] = tokenizer.mask_token_id

            all_labels.append(labels)
            all_input_ids.append(input_ids)
            # print('Final ids: ', input_ids)
            # print('Final labels: ', labels)

        # Stack labels and ids
        labels_stack = torch.stack(all_labels)
        input_ids_stack = torch.stack(all_input_ids)

        return input_ids_stack, labels_stack

    def training_step(self, batch, batch_idx):
        # TODO: Mask for probe train with
        inputs, labels = self.probe_mask_tokens(
            batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]

        tensorboard_logs = {'training_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_idx):
    #     inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

    #     loss = self.forward(inputs, masked_lm_labels=labels,
    #                         layer=self.hparams.probing_layer)[0]
    #     self.log('test_loss', loss)

    # def validation_step(self, batch, batch_idx):
    #     inputs, labels = self.mask_tokens(
    #         batch, self.tokenizer, self.hparams)

    #     loss = self.forward(inputs, masked_lm_labels=labels,
    #                         layer=self.hparams.probing_layer)[0]
    #     self.log('val_loss', loss)

    def train_dataloader(self):
        print('Using my train dataloader')
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.hparams.probing_batch_size, collate_fn=self.probe_cloze_collate, pin_memory=False, num_workers=self.hparams.num_workers)

        return train_dataloader

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
