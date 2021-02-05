from json import decoder
from knowledge_probing.models.t5_decoder_head import T5DecoderHead
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


class T5Decoder(BaseDecoder):
    def __init__(self, hparams):
        super(T5Decoder, self).__init__()

        self.hparams = hparams
        self.model, self.config, self.decoder = self.prepare_model(
            hparams=self.hparams)

        if self.hparams.model_type == 'castorini/monot5-base-msmarco':
            self.tokenizer = AutoTokenizer.from_pretrained(
                't5-base', config=self.config)
        else:
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

        # Load a T5 model
        if hparams.use_model_from_dir:
            print('Loading model from dir: ', hparams.model_dir)
            # model = AutoModel.from_pretrained(hparams.model_dir, config=config)
            model = T5ForConditionalGeneration.from_pretrained(
                hparams.model_dir, config=config)
        else:
            print('Loading model: ', hparams.model_type)
            # model = AutoModel.from_pretrained(
            #     hparams.model_type, config=config)
            model = T5ForConditionalGeneration.from_pretrained(
                hparams.model_type, config=config)

        # Make sure the T5 model itself is not being trained
        model.eval()
        model.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

        # Load the MLM-head
        if hparams.decoder_initialization == 'pre-trained':
            print('Using pre-trained decoder initialization')
            decoder = T5DecoderHead(config, pre_trained_lm_layer=model.lm_head)
        else:
            print('Using random decoder initialization')
            decoder = T5DecoderHead(config)

        # Set mode (encoder/decoder training and probing)
        self.probe_encoder = False
        num_layers = config.num_layers * 2
        encoder_layers = int(num_layers / 2)
        if hparams.probing_layer <= encoder_layers:
            self.probe_encoder = True

        print('***************************************    END    **********************************************')

        return model, config, decoder

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

    def old_forward(self, inputs, masked_lm_labels, t5_labels=None, attention_mask=None, layer=None):

        # get attention mask
        if attention_mask == None:
            attention_mask = inputs.clone()
            attention_mask[attention_mask != self.tokenizer.pad_token_id] = 1
            attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0

        og_labels = None

        # get the correct embedding for the specified layer
        # decoder layers handle tokens differently
        num_layers = self.config.num_layers * 2
        encoder_layers = int(num_layers / 2)
        if self.probe_encoder:
            # Use encoder
            # Set decoder_input_ids to input_ids if you do not require them as per: https://huggingface.co/transformers/model_doc/t5.html#tft5model
            # model_outputs = self.model(
            #     inputs, attention_mask=attention_mask, decoder_input_ids=inputs)
            model_outputs = self.model(
                inputs, attention_mask=attention_mask, labels=inputs)
            embeddings = model_outputs['encoder_hidden_states'][layer]

            # only for debugging:
            og_labels = masked_lm_labels.clone()
        else:
            # Use decoder
            # Set the decoder inputs to the constructed t5-labels if we do decoder

            decoder_attention_mask = t5_labels.clone()
            decoder_attention_mask = self._shift_right(decoder_attention_mask)
            decoder_attention_mask[decoder_attention_mask !=
                                   self.tokenizer.pad_token_id] = 1
            decoder_attention_mask[decoder_attention_mask ==
                                   self.tokenizer.pad_token_id] = 0

            model_outputs = self.model(
                inputs, attention_mask=attention_mask, labels=t5_labels, decoder_attention_mask=decoder_attention_mask)
            decoder_layer = layer - encoder_layers
            embeddings = model_outputs['decoder_hidden_states'][decoder_layer]
            # Mask the t5 labels so that the <extra_id_0> tokens are -100 except for answer tokens these should be the GT answer

            # print("Model inputs: ")
            # print('Input_ids: ', self.tokenizer.decode(inputs[0]))
            # print('Labels: ', self.tokenizer.decode(t5_labels[0]))
            # print('Len embeddings: ', len(embeddings))
            # print('Embeddings: ', embeddings.shape)
            # print(embeddings)

            og_labels = t5_labels.clone()
            # print('OG_labels: ', og_labels)
            # og_labels = self._shift_right(og_labels)

            masked_lm_labels = t5_labels

            # TODO: Potentially shift labels right?
            # masked_lm_labels = self._shift_right(masked_lm_labels)

            # masked_lm_labels[masked_lm_labels > 32000] = -100
            # Just in case also mask the pad tokens (0)
            # TODO: This was actually in
            masked_lm_labels[masked_lm_labels ==
                             self.tokenizer.pad_token_id] = -100

        # Feed embeddings into decoder
        # print('T5Decoder input: ')
        # print('embeddings: ', embeddings)
        # print('MLM labels: ', masked_lm_labels)

        # print('Decoder label input: ', masked_lm_labels)

        decoder_outputs = self.decoder(
            embeddings, labels=masked_lm_labels)

        loss = decoder_outputs[0]
        prediction_scores = decoder_outputs[1]

        # Print information
        # print('Input sentence: ', self.tokenizer.decode(inputs[0], clean_up_tokenization_spaces=False))
        # print('decoder input labels: ', masked_lm_labels[0])
        # print('T5-labels: ', self.tokenizer.decode(masked_lm_labels[0]))
        # print('T5')
        self.total_num_training_steps = self.total_num_training_steps + 1

        if self.total_num_training_steps % 90 == 0:
            # Print some statistics
            # Print information
            print('Encoder decoder mode: probe_encoder: ', self.probe_encoder)
            print('Input sentence: ', self.tokenizer.decode(
                inputs[0], clean_up_tokenization_spaces=False))
            print('decoder input labels: ', masked_lm_labels[0])
            mlm_labels = masked_lm_labels[0].clone()
            mlm_labels[mlm_labels ==
                       -100] = self.tokenizer.pad_token_id
            try:
                print('T5-labels: ',
                      self.tokenizer.decode(og_labels[0]).replace('<pad>', ''))
                # print('T5')
                # mask_idx = 1
                if self.probe_encoder:
                    mask_idx = self.get_index_for_masked_token(
                        inputs[0].cpu(), masked_lm_labels[0].cpu(), mask_token='<extra_id_10>')
                    print('Found an encoder mask index: ', mask_idx)
                else:
                    mask_idx = self.get_index_for_masked_token(
                        inputs[0].cpu(), og_labels[0].cpu(), mask_token='<extra_id_10>')
                # print('Mask token at index: ', mask_idx)
                print('Ground-truth answer: ',
                      self.tokenizer.decode([masked_lm_labels[0][mask_idx]]))
                # print('Top predictions for the masked token: ', topk(
                #     prediction_scores, mask_idx+1, k=10, tokenizer=self.tokenizer))
                print('Top predictions for the masked token: ', topk(
                    prediction_scores, mask_idx, k=10, tokenizer=self.tokenizer))
                # print('Top predictions for the masked token + 1: ', topk(
                #     prediction_scores, mask_idx+1, k=10, tokenizer=self.tokenizer))
                # print('Top predictions for the masked token - 1: ', topk(
                #     prediction_scores, mask_idx-1, k=10, tokenizer=self.tokenizer))
                # print('Decoder loss: ', loss)
                # print('OG T5 loss: ', model_outputs.loss)
                print('OG T5 predictions mask: ', topk(
                    model_outputs[1], mask_idx, k=10, tokenizer=self.tokenizer))
                # print('OG T5 predictions mask + 1: ', topk(
                #     model_outputs[1], mask_idx+1, k=10, tokenizer=self.tokenizer))
                # print('OG T5 predictions mask - 1: ', topk(
                #     model_outputs[1], mask_idx-1, k=10, tokenizer=self.tokenizer))
            except:
                print('No label found, therefore no output')

        return loss, prediction_scores

    def forward(self, input_ids, masked_lm_labels, t5_labels, layer, attention_mask=None):
        # Get encoder attention mask from input_ids
        if attention_mask == None:
            encoder_attention_mask = self.get_attention_mask(input_ids)
        else:
            encoder_attention_mask = attention_mask

        initial_t5_labels = t5_labels.clone()
        # initial_t5_labels = t5_labels

        # Encode
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            output_hidden_states=self.config.output_hidden_states
        )

        if self.probe_encoder:
            all_hidden_states_encoder = encoder_outputs.hidden_states
            embeddings = all_hidden_states_encoder[layer]
            labels = masked_lm_labels
        else:
            # Get the correct layer to probe
            num_layers = self.config.num_layers * 2
            encoder_layers = int(num_layers / 2)
            decoder_layer = layer - encoder_layers

            # Get decoder attention mask
            decoder_attention_mask = self.get_attention_mask(t5_labels)

            # Shift embeddings for decoder inputs
            decoder_input_ids = self._shift_right(t5_labels)

            # Decode
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=encoder_attention_mask,
                output_hidden_states=self.config.output_hidden_states
            )

            all_hidden_states_decoder = decoder_outputs[2]
            embeddings = all_hidden_states_decoder[decoder_layer]

            labels = initial_t5_labels
            labels[labels > 32000] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

        decoder_outputs = self.decoder(
            embeddings, labels=labels)

        loss = decoder_outputs[0]
        prediction_scores = decoder_outputs[1]

        # Optional printing
        self.total_num_training_steps = self.total_num_training_steps + 1

        if self.total_num_training_steps % 90 == 0:
            # Print some statistics
            # print('Encoder decoder mode: probe_encoder: ', self.probe_encoder)
            # print('Input sentence: ', self.tokenizer.decode(
            #     input_ids[0], clean_up_tokenization_spaces=False))
            # print('Encoder attention mask: ', encoder_attention_mask[0])
            # mlm_labels = masked_lm_labels[0].clone()
            # mlm_labels[mlm_labels ==
            #            -100] = self.tokenizer.pad_token_id
            try:
                if self.probe_encoder:
                    mask_idx = self.get_index_for_masked_token(
                        input_ids[0].cpu(), masked_lm_labels[0].cpu(), mask_token='<extra_id_10>')
                    # print('Found an encoder mask index: ', mask_idx)
                    # model_outputs = encoder_outputs
                else:
                    # print('Decoder input ids: ', decoder_input_ids[0])
                    # print('Decoder attention mask: ',
                    #       decoder_attention_mask[0])
                    mask_idx = self.get_index_for_masked_token(
                        input_ids[0].cpu(), t5_labels[0].cpu(), mask_token='<extra_id_10>')
                    # print('Found an decoder mask index: ', mask_idx)
                    # model_outputs = decoder_outputs

                # print('Classifying head labels: ', labels[0])

                print('Ground-truth answer: ',
                      self.tokenizer.decode([labels[0][mask_idx]]))

                # Predictions
                print('Top predictions for the masked token: ', topk(
                    prediction_scores, mask_idx, k=10, tokenizer=self.tokenizer))
                # print('Top predictions for the masked token + 1: ', topk(
                #     prediction_scores, mask_idx+1, k=10, tokenizer=self.tokenizer))
                # print('Top predictions for the masked token - 1: ', topk(
                #     prediction_scores, mask_idx-1, k=10, tokenizer=self.tokenizer))
                # print('OG T5 predictions mask: ', topk(
                #     model_outputs[1], mask_idx, k=10, tokenizer=self.tokenizer))

            except:
                print('No label found, therefore no output')

        return loss, prediction_scores

    def get_attention_mask(self, input: torch.Tensor) -> torch.Tensor:
        attention_mask = input.clone()
        attention_mask[attention_mask != self.tokenizer.pad_token_id] = 1
        attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0

        return attention_mask

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
            # print(prediction_scores)
            # print(prediction_scores.shape)

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

        if self.probe_encoder:
            # Use encoder schema: Everything that is not a mask token should be -100
            masked_labels = input_ids_tensor.clone()
            # mask_index = self.get_index_for_masked_token(input_ids=masked_labels)
            masked_labels[masked_labels != self.mask_token_id] = -100
            # print('Masked labels: ', masked_labels)
            return masked_labels
        else:
            # Use decoder schema
            # For the decoder the idea is that the second token should be the correct one as it will be: mask, token
            # txt = '{} {}'.format(self.mask_token, obj_label)
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

        if self.probe_encoder:
            # Use encoder schema
            idx = input_ids.numpy().tolist().index(mask_token_id)
            # print('Encoder schema: ', idx)
            return idx
        else:
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

        # tensorboard_logs = {'training_loss': loss}
        self.log('training_loss', loss, on_step=True)
        return loss

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

    def configure_optimizers(self):
        if self.hparams.use_adafactor:
            adafactor = Adafactor([p for p in self.parameters(
            ) if p.requires_grad], relative_step=self.hparams.adafactor_relative_step, warmup_init=self.hparams.adafactor_warmup, scale_parameter=self.hparams.adafactor_scale_params)  # lr=self.hparams.lr,

            return [adafactor]
        else:
            adam = AdamW([p for p in self.parameters()
                          if p.requires_grad], lr=self.hparams.lr, eps=1e-08)

            scheduler = get_linear_schedule_with_warmup(
                adam, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)

            return [adam], [{"scheduler": scheduler, "interval": "step"}]

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     """
    #     Specify the hyperparams for this LightningModule
    #     """
    #     # MODEL specific
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--use_adafactor',
    #                         default=False, action='store_true')
    #     parser.add_argument('--adafactor_relative_step',
    #                         default=False, action='store_true')
    #     parser.add_argument('--adafactor_warmup',
    #                         default=False, action='store_true')
    #     parser.add_argument('--adafactor_scale_params',
    #                         default=False, action='store_true')
