from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import functools
from typing import Tuple, List
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from knowledge_probing.models.decoder_head import MyDecoderHead
from knowledge_probing.datasets.text_dataset import TextDataset
from knowledge_probing.datasets.text_data_utils import mask_tokens, collate


class Decoder(LightningModule):
    def __init__(self, hparams, bert, config):
        super(Decoder, self).__init__()
        self.decoder = MyDecoderHead(config)

        self.bert = bert
        self.bert.eval()
        self.bert.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.bert_model_type, use_fast=False)
        self.hparams = hparams

        self.collate = functools.partial(collate, tokenizer=self.tokenizer)

    def forward(self, inputs, masked_lm_labels, attention_mask=None, layer=None, all_layers=False):

        # get attention mask
        if attention_mask == None:
            attention_mask = inputs.clone()
            attention_mask[attention_mask != self.tokenizer.pad_token_id] = 1
            attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0

        # get Berts embeddings
        bert_outputs = self.bert(inputs, attention_mask=attention_mask)
        if not all_layers:
            if layer == 12:
                embeddings = bert_outputs[0]
            else:
                embeddings = bert_outputs[2][layer]

            # Feed embeddings into decoder
            decoder_outputs = self.decoder(
                embeddings, masked_lm_labels=masked_lm_labels)
            loss = decoder_outputs[0]
            prediction_scores = decoder_outputs[1]

            return loss, prediction_scores

        if all_layers:
            # Run all embeddings through the decoder and bundle their predictions
            layer_outputs = []

            for i, layer_embeddings in enumerate(bert_outputs[2]):
                if i == 0:
                    # Step over input_embeddings
                    continue
                   # Feed embeddings into decoder
                decoder_outputs = self.decoder(
                    layer_embeddings, masked_lm_labels=masked_lm_labels)
                prediction_scores = decoder_outputs[1]

                layer_outputs.append(prediction_scores)

            return layer_outputs

    def prepare_data(self):
        self.train_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.train_data_file, block_size=self.tokenizer.max_len)
        self.eval_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.eval_data_file, block_size=self.tokenizer.max_len)
        self.test_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.test_data_file, block_size=self.tokenizer.max_len)

    def train_dataloader(self):
        self.hparams.train_batch_size = self.hparams.per_gpu_train_batch_size * 1

        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.hparams.train_batch_size, collate_fn=self.collate)

        return train_dataloader

    def val_dataloader(self):
        self.hparams.eval_batch_size = self.hparams.per_gpu_eval_batch_size * 1

        eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=self.hparams.eval_batch_size, collate_fn=self.collate)

        return eval_dataloader

    def test_dataloader(self):
        self.hparams.test_batch_size = self.hparams.per_gpu_test_batch_size * 1

        test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.hparams.test_batch_size, collate_fn=self.collate)

        return test_dataloader

    def configure_optimizers(self):
        # adam = torch.optim.Adam([p for p in self.decoder.parameters() if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08)
        adam = AdamW([p for p in self.decoder.parameters(
        ) if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08)

        # scheduler = get_linear_schedule_with_warmup(
        #     adam, num_warmup_steps=hparams.warmup_steps, num_training_steps=(hparams.training_epochs * 100)
        # )

        scheduler = ReduceLROnPlateau(adam, mode='min')

        return [adam], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]

        tensorboard_logs = {'training_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        perplexity = torch.exp(avg_loss)

        tensorboard_logs = {
            'avg_test_loss': avg_loss, 'perplexity': perplexity}
        return {"avg_test_loss": avg_loss, 'perplexity': perplexity, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        perplexity = torch.exp(avg_loss)
        print(perplexity)

        tensorboard_logs = {'avg_val_loss': avg_loss, 'perplexity': perplexity}
        return {"val_loss": avg_loss, 'perplexity': perplexity, "log": tensorboard_logs}
