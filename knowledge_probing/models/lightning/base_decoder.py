from __future__ import annotations
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from knowledge_probing.datasets.text_dataset import TextDataset
from knowledge_probing.datasets.full_wiki_text_dataset import FullWikiTextDataset
from argparse import ArgumentParser, Namespace
from knowledge_probing.models.bert_model_util import mask_tokens
from knowledge_probing.file_utils import find_checkpoint_in_dir
from torch.nn.utils.rnn import pad_sequence
import sys


class BaseDecoder(LightningModule):
    def __init__(self):
        super(BaseDecoder, self).__init__()

        self.mask_token = 'pass'
        self.mask_token_id = 'pass'

    def load_best_model_checkpoint(self, hparams: Namespace):
        checkpoint_file = find_checkpoint_in_dir(hparams.decoder_save_dir)

        print('Loading best checkpoint: {}'.format(checkpoint_file))

        # After normal initialization, this will override the models state dicts with the saved ones.
        best_model = self.load_from_checkpoint(checkpoint_file)

        return best_model

    def forward(self, inputs: torch.Tensor, masked_lm_labels: torch.Tensor, attention_mask: torch.Tensor = None, layer: int = None):
        pass

    def probe(self, batch, layer: int, relation_args: Namespace):
        pass

    def set_to_train(self):
        """
        This function is intended to set everthing of the model to training mode (given hparams.unfreeze_transformer=True). 
        Otherwise, we only set the decoder to train.
        """
        pass

    def set_to_eval(self):
        """
        This function is intended to set everthing of the model to evaluation mode. 
        """
        pass

    def mlm_collate(self, examples, tokenizer=None):
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    def cloze_collate(self, examples, tokenizer: AutoTokenizer):
        """ 
            This is a function that makes sure all entries in the batch are padded 
            to the correct length.
        """
        masked_sentences = [x['masked_sentences'] for x in examples]
        uuids = [x['uuid'] for x in examples]
        obj_labels = [x['obj_label'] for x in examples]
        mask_indices = [x['mask_index'] for x in examples]

        padded_sentences = pad_sequence(
            masked_sentences, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = padded_sentences.clone()
        attention_mask[attention_mask != tokenizer.pad_token_id] = 1
        attention_mask[attention_mask == tokenizer.pad_token_id] = 0

        examples_batch = {
            "masked_sentences": padded_sentences,
            "attention_mask": attention_mask,
            "obj_label": obj_labels,
            "uuid": uuids,
            "mask_index": mask_indices
        }

        if 'judgments' in examples[0]:
            examples_batch['judgments'] = [x['judgments'] for x in examples]

        return examples_batch

    def get_index_for_masked_token(self, tensor: torch.Tensor):
        return tensor.numpy().tolist().index(self.mask_token_id)

    def prepare_data(self):
        if self.hparams.use_full_wiki:
            self.train_dataset = FullWikiTextDataset(
                self.tokenizer, self.hparams, block_size=self.tokenizer.model_max_length, mode='train')
            self.eval_dataset = FullWikiTextDataset(
                self.tokenizer, self.hparams, block_size=self.tokenizer.model_max_length, mode='valid')
            self.test_dataset = FullWikiTextDataset(
                self.tokenizer, self.hparams, block_size=self.tokenizer.model_max_length, mode='test')
        else:
            self.train_dataset = TextDataset(
                self.tokenizer, self.hparams, file_path=self.hparams.train_file, block_size=self.tokenizer.model_max_length)
            self.eval_dataset = TextDataset(
                self.tokenizer, self.hparams, file_path=self.hparams.valid_file, block_size=self.tokenizer.model_max_length)
            self.test_dataset = TextDataset(
                self.tokenizer, self.hparams, file_path=self.hparams.test_file, block_size=self.tokenizer.model_max_length)

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collate, pin_memory=False, num_workers=self.hparams.num_workers)

        # self.total_num_training_steps = self.hparams.max_epochs * \
        #     len(train_dataloader)

        return train_dataloader

    def val_dataloader(self):
        eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collate, pin_memory=False, num_workers=self.hparams.num_workers)

        return eval_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collate, pin_memory=False, num_workers=self.hparams.num_workers)

        return test_dataloader

    def configure_optimizers(self):
        adam = AdamW([p for p in self.parameters(
        ) if p.requires_grad], lr=self.hparams.lr, eps=1e-08)

        # print('Configuring optimizer, total number steps: ',
        #       self.total_num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            adam, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)

        # scheduler = ReduceLROnPlateau(adam, mode='min')

        return [adam], [{"scheduler": scheduler, "interval": "step"}]

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
        self.log('test_loss', loss)

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     perplexity = torch.exp(avg_loss)

    #     tensorboard_logs = {
    #         'avg_test_loss': avg_loss, 'perplexity': perplexity}
    #     return {"avg_test_loss": avg_loss, 'perplexity': perplexity, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]
        self.log('val_loss', loss)

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     perplexity = torch.exp(avg_loss)
    #     print(perplexity)

    #     tensorboard_logs = {'avg_val_loss': avg_loss, 'perplexity': perplexity}
    #     return {"val_loss": avg_loss, 'perplexity': perplexity, "log": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', default=0.02, type=float)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--warmup_steps', default=5000, type=int,
                            help='Number of steps that is used for the linear warumup')
        parser.add_argument('--total_steps', default=60000, type=int,
                            help='Number of steps that is used for the decay after the warumup')

        parser.add_argument('--num_workers', default=4, type=int,
                            help='Number of cpu workers used by the dataloaders')

        parser.add_argument('--mlm_probability', default=0.15, type=float)

        parser.add_argument('--use_model_from_dir',
                            default=False, action='store_true')
        parser.add_argument(
            '--model_dir', required='--use_model_from_dir' in sys.argv)
        parser.add_argument('--model_type', default='bert-base-uncased',
                            choices=['bert-base-uncased', 'bert-base-cased', 't5-small', 't5-base',
                                     'google/t5-v1_1-small', 'google/t5-v1_1-base', 'castorini/monot5-base-msmarco', 'valhalla/t5-base-squad'])
        parser.add_argument('--unfreeze_transformer',
                            default=False, action='store_true')
        parser.add_argument('--use_full_wiki',
                            default=False, action='store_true')
        parser.add_argument('--full_wiki_cache_dir',
                            required='--use_full_wiki' in sys.argv)
        parser.add_argument(
            '--train_file', default="data/training_data/wikitext-2-raw/wiki.train.raw")
        parser.add_argument(
            '--valid_file', default="data/training_data/wikitext-2-raw/wiki.valid.raw")
        parser.add_argument(
            '--test_file', default="data/training_data/wikitext-2-raw/wiki.test.raw")

        parser.add_argument('--probing_data_dir', default="data/probing_data/")
        parser.add_argument('--probing_batch_size', default=16, type=int)
        parser.add_argument('--precision_at_k', default=100, type=int,
                            help='When probing, we compute precision at 1, 10, and k. Feel free to set the k here')

        parser.add_argument('--use_adafactor',
                            default=False, action='store_true')
        parser.add_argument('--adafactor_relative_step',
                            default=False, action='store_true')
        parser.add_argument('--adafactor_warmup',
                            default=False, action='store_true')
        parser.add_argument('--adafactor_scale_params',
                            default=False, action='store_true')

        return parser
