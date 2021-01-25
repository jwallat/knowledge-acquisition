from knowledge_probing.models.lightning.t5_decoder import T5Decoder
from knowledge_probing.datasets.text_data_utils import mask_tokens
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
import sys
import torch
import functools
from argparse import ArgumentParser, Namespace
from transformers.optimization import Adafactor
from typing import Tuple, List
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig, AutoModel, T5ForConditionalGeneration
from knowledge_probing.probing.metrics import calculate_metrics
from knowledge_probing.models.decoder_head import MyDecoderHead
from knowledge_probing.datasets.text_dataset import TextDataset
from knowledge_probing.datasets.cloze_data_utils import topk
from knowledge_probing.models.t5_model_util import mask_tokens


class OGT5Model(BaseDecoder):
    def __init__(self, hparams):
        super(OGT5Model, self).__init__()

        self.hparams = hparams
        self.model, self.config = self.prepare_model(
            hparams=self.hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_type, config=self.config)

        self.total_num_training_steps = 0
        self.collate = functools.partial(
            self.mlm_collate, tokenizer=self.tokenizer)

        self.mask_token = '<extra_id_0>'
        self.mask_token_id = 32099

    def prepare_model(self, hparams: Namespace):
        # print(hparams)
        config = AutoConfig.from_pretrained(hparams.model_type)
        config.output_hidden_states = True

        print('***********************************    LOADING MODEL    ****************************************')

        print('The model {} has {} encoder and {} decoder layers'.format(
            hparams.model_type, config.num_layers, config.num_layers))

        model = T5ForConditionalGeneration.from_pretrained(
            hparams.model_type, config=config)

        # Set model to eval
        model.eval()
        model.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

        print('***************************************    END    **********************************************')

        return model, config

    def set_to_eval(self):
        """
        This function is intended to set everthing of the model to evaluation mode. 
        """
        self.eval()
        self.requires_grad = False
        for param in self.parameters():
            param.requires_grad = False
        print('Everything set to evaluation')

    def forward(self, inputs, masked_lm_labels, t5_labels, attention_mask=None, layer=None):

        # get attention mask
        if attention_mask == None:
            attention_mask = inputs.clone()
            attention_mask[attention_mask != self.tokenizer.pad_token_id] = 1
            attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0

        # decoder_attention_mask = None
        decoder_attention_mask = t5_labels.clone()
        # decoder_attention_mask = self._shift_right(decoder_attention_mask)
        decoder_attention_mask[decoder_attention_mask !=
                               self.tokenizer.pad_token_id] = 1
        decoder_attention_mask[decoder_attention_mask ==
                               self.tokenizer.pad_token_id] = 0

        # print(100*"-" + '    Model inputs    ' + 100*"-")
        # # print('Input sentence: ', self.tokenizer.decode(
        # #     inputs[0], clean_up_tokenization_spaces=False))
        # print('Inputs: ', inputs)
        # # print('T5 labels: ', t5_labels[0])
        # print('T5-labels: ',
        #       self.tokenizer.decode(t5_labels[0]).replace('<pad>', ''))
        # print("Attention mask: ", attention_mask)
        # print('T5-labels: ',
        #         self.tokenizer.decode(t5_labels[0]).replace('<pad>', ''))

        outputs = self.model(
            input_ids=inputs, labels=t5_labels, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)

        loss = outputs.loss
        prediction_scores = outputs.logits

        try:
            mask_idx = self.get_index_for_masked_token(
                inputs[0].cpu(), t5_labels[0].cpu(), mask_token='<extra_id_0>')
            print('Ground-truth answer: ',
                  self.tokenizer.decode([t5_labels[0][mask_idx]]))
            print('Top predictions for the masked token: ', topk(
                prediction_scores, mask_idx, k=10, tokenizer=self.tokenizer))
        except:
            print('Couldn\'t print')

        return loss, prediction_scores

    def probe(self, batch, layer, relation_args):
        '''
        Wrapper for the forward method that is used when probing. 
        It should encapsulate all the model specific behavior away from the probing loop.
        '''

        # print(batch)

        input_ids_batch = batch['masked_sentences']
        input_ids_batch = input_ids_batch.to(self.hparams.device)
        attention_mask_batch = batch['attention_mask']
        attention_mask_batch = attention_mask_batch.to(self.hparams.device)

        # Get the T5 labels/index for the correct layer
        t5_labels_batch = batch['t5_labels']
        t5_labels_batch = t5_labels_batch.to(self.hparams.device)
        mask_indices_batch = batch['mask_index']
        # mask_indices_batch = mask_indices_batch.to(self.hparams.device)

        # Get model's prediction scores
        outputs = self.forward(input_ids_batch, masked_lm_labels=t5_labels_batch,
                               t5_labels=t5_labels_batch, attention_mask=attention_mask_batch, layer=layer)

        batch_prediction_scores = outputs[1]

        metrics_elements_from_batch = []
        for i, prediction_scores in enumerate(batch_prediction_scores):

            prediction_scores = prediction_scores[None, :, :]

            metrics_element = calculate_metrics(
                batch, i, prediction_scores, precision_at_k=relation_args.precision_at_k, tokenizer=self.tokenizer)
            metrics_elements_from_batch.append(metrics_element)

        # For elements in batch, compute the metrics elements
        return metrics_elements_from_batch

    def get_probing_t5_labels(self, input_ids_tensor, obj_label):
        '''
        Function for producing t5_labels for the probing task. Here we distinguish between encoder and decoder.
        '''
        # TODO: Test if a different label will actually cause the model to fail. Currently, it looks to good to be true.
        txt = '{} {} {}'.format(self.mask_token, obj_label, '<extra_id_1>')
        t5_labels = self.tokenizer.encode(txt, return_tensors='pt')[0]
        # print(t5_labels)

        return t5_labels

    def get_index_for_masked_token(self, input_ids: torch.Tensor, t5_labels: torch.Tensor = None, mask_token: str = None):
        '''
        Helper function that will return the index of the masked token in the final prediction scored. 
        For T5 this is a bit tricky as the returned prediction scores differ between encoder and decoder layers: 
        Where in encoder layers, the original sentence is returned, the decoder layers only return prediction scores for a set of 
        [<extra_id_X> _answer_token_extra_id_x_]. 
        '''
        if mask_token:
            mask_token_id = self.tokenizer.encode(mask_token)[0]
        else:
            mask_token_id = self.mask_token_id
        # print('Index for this input: ', input_ids)
        # if self.probe_encoder:
        #     # Use encoder schema
        #     idx = input_ids.numpy().tolist().index(self.mask_token_id)
        #     # print('Encoder schema: ', idx)
        #     return idx
        # else:
        # Use decoder schema
        # For the decoder the idea is that the second token should be the correct one as it will be: mask, token
        mask_token_index = t5_labels.numpy().tolist().index(mask_token_id)
        index_of_answer_token = mask_token_index + 1
        # print('Decoder schema: ', index_of_answer_token)
        return index_of_answer_token

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(
        ), "Verify that `shifted_input_ids` has only positive values"

        # print('Shifted: ', shifted_input_ids)
        return shifted_input_ids

    def cloze_collate(self, examples, tokenizer: AutoTokenizer):
        """ 
            This is a function that makes sure all entries in the batch are padded 
            to the correct length.
        """
        masked_sentences = [x['masked_sentences'] for x in examples]
        uuids = [x['uuid'] for x in examples]
        obj_labels = [x['obj_label'] for x in examples]
        mask_indices = [x['mask_index'] for x in examples]

        t5_labels = [x['t5_labels'] for x in examples]
        t5_labels = pad_sequence(
            t5_labels, batch_first=True, padding_value=tokenizer.pad_token_id)

        padded_sentences = pad_sequence(
            masked_sentences, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = padded_sentences.clone()
        attention_mask[attention_mask != tokenizer.pad_token_id] = 1
        attention_mask[attention_mask == tokenizer.pad_token_id] = 0

        examples_batch = {
            "masked_sentences": padded_sentences,
            "attention_mask": attention_mask,
            't5_labels': t5_labels,
            "obj_label": obj_labels,
            "uuid": uuids,
            "mask_index": mask_indices
        }

        if 'judgments' in examples[0]:
            examples_batch['judgments'] = [x['judgments'] for x in examples]

        return examples_batch

    # Optionally, overwrite any lightning training functions or cli arugments (make sure to add them in main function!)

    def training_step(self, batch, batch_idx):
        inputs, labels, t5_labels = mask_tokens(
            batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels, t5_labels=t5_labels,
                            layer=self.hparams.probing_layer)[0]

        tensorboard_logs = {'training_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels, t5_labels = mask_tokens(
            batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels, t5_labels=t5_labels,
                            layer=self.hparams.probing_layer)[0]
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        inputs, labels, t5_labels = mask_tokens(
            batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels, t5_labels=t5_labels,
                            layer=self.hparams.probing_layer)[0]
        self.log('test_loss', loss)

    # def configure_optimizers(self):
    #     adafactor = Adafactor([p for p in self.parameters(
    #     ) if p.requires_grad], relative_step=True, warmup_init=True, scale_parameter=True)  # lr=self.hparams.lr,

    #     # print('Configuring optimizer, total number steps: ',
    #     #       self.total_num_training_steps)
    #     # scheduler = get_linear_schedule_with_warmup(
    #     #     adam, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)

    #     # scheduler = ReduceLROnPlateau(adam, mode='min')

    #     return [adafactor]

    def configure_optimizers(self):
        adam = AdamW([p for p in self.parameters()
                      if p.requires_grad], lr=self.hparams.lr, eps=1e-08)

        scheduler = get_linear_schedule_with_warmup(
            adam, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)

        return [adam], [{"scheduler": scheduler, "interval": "step"}]
