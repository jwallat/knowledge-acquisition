from json import decoder
from knowledge_probing.models.t5_decoder_head import T5DecoderHead
# from knowledge_probing.datasets.text_data_utils import mask_tokens
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
import sys, copy
import torch
import functools
from argparse import ArgumentParser, Namespace
from transformers.optimization import Adafactor
from typing import Tuple, List
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig, AutoModel, \
    T5ForConditionalGeneration
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from knowledge_probing.probing.metrics import calculate_metrics, calculate_metrics4paq
from knowledge_probing.models.decoder_head import MyDecoderHead
from knowledge_probing.datasets.text_dataset import TextDataset
from knowledge_probing.datasets.cloze_data_utils import topk
from knowledge_probing.models.t5_model_util import mask_tokens, old_mask_tokens, qa_tokens, ssm_tokens, pmi_tokens
import json, os
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
from knowledge_probing.file_utils import find_checkpoint_in_dir
from typing import Optional, Callable
from torch.optim.optimizer import Optimizer
import random
import pickle5 as pickle


class T5Decoder(BaseDecoder):

    def __init__(self, hparams):
        super(T5Decoder, self).__init__()

        self.save_hyperparameters(hparams)  # self.hparams = hparams
        self.model, self.config = self.prepare_model(
            hparams=self.hparams)

        if self.hparams.model_type == 'castorini/monot5-base-msmarco':
            self.tokenizer = AutoTokenizer.from_pretrained(
                't5-base', config=self.config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.model_type, config=self.config)

        self.total_num_training_steps = 0
        self.total_num_probing_steps = 0
        if self.hparams.mask_way == 'ssm' or self.hparams.mask_way == 'pmi':
            self.collate = self.ssm_collate
        else:
            self.collate = functools.partial(self.mlm_collate, tokenizer=self.tokenizer)
            # if self.hparams.mask_way == 'pmi':
            #     self.pmi = self.prepare_pmi(pmi_path=self.hparams.pmi_path)
        if self.hparams.ewc:
            self.ewc = self.prepare_ewc()
            self.ewc_loss = torch.tensor(0)
        self.mask_token = '<extra_id_0>'
        self.mask_token_id = 32099

    def prepare_model(self, hparams: Namespace):
        # print(hparams)
        if self.hparams.use_raw_model:
            print('*********************************    LOADING RAW MODEL    **************************************')
            config = T5Config(vocab_size=32128, d_model=768, d_kv=64, d_ff=3072, num_layers=12, num_decoder_layers=12,
                              num_heads=12, relative_attention_num_buckets=32, dropout_rate=0.05,
                              layer_norm_epsilon=1e-6, initializer_factor=1.0, feed_forward_proj="relu",
                              is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1,
                              decoder_start_token_id=0,
                              n_positions=512, )
            model = T5ForConditionalGeneration(config)
        else:
            config = AutoConfig.from_pretrained(hparams.model_type)
            config.output_hidden_states = True

            print('***********************************    LOADING MODEL    ****************************************')

            print('The model {} has {} encoder and {} decoder layers'.format(
                hparams.model_type, config.num_layers, config.num_layers))

            # Load a T5 model
            if hparams.use_model_from_dir:
                print('Loading model from dir: ', hparams.model_dir)
                model = T5ForConditionalGeneration.from_pretrained(
                    hparams.model_dir, config=config)
            else:
                print('Loading model: ', hparams.model_type)
                model = T5ForConditionalGeneration.from_pretrained(
                    hparams.model_type, config=config)

        num_decoder_layer = hparams.probing_layer - 12
        model.decoder.block = model.decoder.block[:num_decoder_layer]
        if self.hparams.load_model_ckpt_path:
            if 'ckpt' not in self.hparams.load_model_ckpt_path:
                checkpoint_file = find_checkpoint_in_dir(self.hparams.load_model_ckpt_path)
            else:
                checkpoint_file = self.hparams.load_model_ckpt_path
            print('Loading last task model checkpoint: {}'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
            model_state_dict = {k.split('model.')[1]: v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(model_state_dict)
        # Make sure the T5 model itself is not being trained
        model.eval()
        model.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

        self.probe_encoder = False
        num_layers = config.num_layers * 2
        encoder_layers = int(num_layers / 2)
        if hparams.probing_layer <= encoder_layers:
            self.probe_encoder = True

        print('***************************************    END    **********************************************')

        return model, config  # , decoder

    def prepare_pmi(self, pmi_path):
        assert os.path.exists(pmi_path)
        data = []
        if 'jsonl' in pmi_path:
            with open(pmi_path) as obj:
                for line in obj.readlines():
                    data.append(json.loads(line))
            return data[0]['pmi']  # , data[1]['test'], data[2]['train']
        else:
            with open(pmi_path, "rb") as handle:
                return pickle.load(handle)[:800000]

    def prepare_ewc(self):
        old_train_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.old_dataset4ewc,
            block_size=self.tokenizer.model_max_length).examples
        #old_train_dataset = random.sample(old_train_dataset, 50)
        print(len(old_train_dataset), type(old_train_dataset))
        ewc = EWC(self.model, old_train_dataset, self.hparams, self.tokenizer)
        print('Prepare EWC ready!')
        return ewc

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            _loss = self.ewc.precision_matrices[n] * (p - self.ewc.means[n]) ** 2
            loss += _loss.sum()

        return loss.detach()

    def set_to_train(self):
        """
        This function is intended to set everthing of the model to training mode (given hparams.unfreeze_transformer=True).
        Otherwise, we only set the decoder to train.
        """
        # self.decoder.train()
        # self.decoder.requires_grad = True
        # for param in self.decoder.parameters():
        #     param.requires_grad = True
        # print('Decoder set to train')
        self.model.train()
        self.model.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = True
        if self.hparams.freeze_decoder:
            self.model.decoder.eval()
            self.model.decoder.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            print('Decoder of T5 was frozen')
        if self.hparams.freeze_encoder:
            self.model.encoder.eval()
            self.model.encoder.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print('Encoder of T5 was frozen')
        if (not self.hparams.freeze_decoder) and (not self.hparams.freeze_encoder):
            print('Whole T5 set to train')

    def set_to_finetuning(self):
        # self.decoder.train()
        # self.decoder.requires_grad = True
        # for param in self.decoder.parameters():
        #     param.requires_grad = True
        # print('Decoder set to fine_tuning')

        self.model.train()
        self.model.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = True
        print("→ → → Whole T5 set to fine-tuning")

        # self.model.encoder.eval()
        # self.model.encoder.requires_grad = False
        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False
        # print("→ → → but T5's encoder set to freezing")

        # self.model.lm_head.train()
        # self.model.requires_grad = True
        # for param in self.model.lm_head.parameters():
        #     param.requires_grad = True
        # print("→ → → T5's lm-head to fine_tuning")

    def set_to_eval(self):
        """
        This function is intended to set everthing of the model to evaluation mode. 
        """
        self.eval()
        self.requires_grad = False
        for param in self.parameters():
            param.requires_grad = False
        print('Everything set to evaluation')

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, multitask_sign=None, mode=None):
        outputs = self.model(input_ids=inputs, labels=labels)
        loss = outputs[0]
        if self.hparams.ewc and self.hparams.multitask:
            if multitask_sign == 'QA':
                loss = loss + self.ewc_loss.to(loss.device)
        elif self.hparams.ewc and mode == 'train':
            loss = loss + self.ewc_loss.to(loss.device)
        return loss

    def get_attention_mask(self, input: torch.Tensor) -> torch.Tensor:

        attention_mask = input.clone()
        attention_mask[attention_mask != self.tokenizer.pad_token_id] = 1
        attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0

        return attention_mask

    def probe(self, batch, layer, relation_args, show=False):
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
                               t5_labels=t5_labels_batch, attention_mask=attention_mask_batch, layer=layer, show=show)

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

    def probe4paq(self, batch, layer, show, log_path):
        input_ids_batch = batch['inputs_id']
        labels_batch = batch['labels']
        t5_labels_batch = batch['t5_labels']
        attention_mask_batch = batch['attention_mask']
        input_ids_batch = input_ids_batch.to(self.hparams.device)
        labels_batch = labels_batch.to(self.hparams.device)
        t5_labels_batch = t5_labels_batch.to(self.hparams.device)
        outputs = self.forward(input_ids_batch, masked_lm_labels=labels_batch, t5_labels=t5_labels_batch,
                               attention_mask=attention_mask_batch, layer=layer, show=show)

        batch_prediction_scores = outputs[1]

        metrics_elements_from_batch = []
        for i, prediction_scores in enumerate(batch_prediction_scores):
            prediction_scores = prediction_scores[None, :, :]
            metrics_element = calculate_metrics4paq(
                batch, i, prediction_scores, precision_at_k=self.hparams.precision_at_k
                , tokenizer=self.tokenizer, log_path=log_path)
            metrics_elements_from_batch.append(metrics_element)

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
            # print(txt)
            t5_labels = self.tokenizer.encode(txt, return_tensors='pt')[0]
            # print(self.tokenizer.decode(t5_labels))
            # print(t5_labels)

            return t5_labels

    def get_index_for_masked_token(self, input_ids: torch.Tensor, t5_labels: torch.Tensor = None,
                                   mask_token: str = None):
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
        # print('---decoder_start_token_id:', decoder_start_token_id)
        pad_token_id = self.config.pad_token_id
        # print('---pad_token_id:', pad_token_id)
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
        # print(batch)
        # print(len(batch))
        if self.hparams.finetuning:
            inputs, t5_labels = qa_tokens(batch, self.tokenizer, self.hparams)
            loss = self.forward(inputs, t5_labels, mode='train')
        elif self.hparams.multitask:
            if batch[0] == 'QA':
                inputs, t5_labels = qa_tokens(batch[-1], self.tokenizer, self.hparams)
            elif batch[0] == 'text':
                if self.hparams.mask_way == 'ssm':
                    inputs = batch[1]
                    t5_labels = batch[2]
                    # inputs, t5_labels = ssm_tokens(batch[-1], self.tokenizer, self.hparams)
                elif self.hparams.mask_way == 'pmi':
                    inputs = batch[1]
                    t5_labels = batch[2]
                    # inputs, t5_labels = pmi_tokens(batch[-1], self.tokenizer, self.hparams, self.pmi)
                else:
                    inputs, labels, t5_labels = mask_tokens(batch[-1], self.tokenizer, self.hparams, self.mask_token)
            loss = self.forward(inputs, t5_labels, multitask_sign=batch[0])
        else:
            if self.hparams.mask_way == 'ssm':
                # inputs, t5_labels = ssm_tokens(batch, self.tokenizer, self.hparams)
                inputs = batch[0]
                t5_labels = batch[1]
            elif self.hparams.mask_way == 'pmi':
                inputs = batch[0]
                t5_labels = batch[1]
                # inputs, t5_labels = pmi_tokens(batch, self.tokenizer, self.hparams, self.pmi)
            else:
                inputs, labels, t5_labels = mask_tokens(batch, self.tokenizer, self.hparams, self.mask_token)
            loss = self.forward(inputs, t5_labels)
        self.log('training_loss', loss)  # , on_step=True
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.finetuning:
            inputs, t5_labels = qa_tokens(batch, self.tokenizer, self.hparams)
            loss = self.forward(inputs, t5_labels)
        else:
            if self.hparams.mask_way == 'ssm':
                # inputs, t5_labels = ssm_tokens(batch, self.tokenizer, self.hparams)
                inputs = batch[0]
                t5_labels = batch[1]
            elif self.hparams.mask_way == 'pmi':
                inputs = batch[0]
                t5_labels = batch[1]
                # inputs, t5_labels = pmi_tokens(batch, self.tokenizer, self.hparams, self.pmi)
            else:
                inputs, labels, t5_labels = mask_tokens(batch, self.tokenizer, self.hparams, self.mask_token)

            loss = self.forward(inputs, t5_labels)

        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        if self.hparams.finetuning:
            inputs, t5_labels = qa_tokens(batch, self.tokenizer, self.hparams)
            loss = self.forward(inputs, t5_labels)
        else:
            if self.hparams.mask_way == 'ssm':
                inputs = batch[0]
                t5_labels = batch[1]
                # inputs, t5_labels = ssm_tokens(batch, self.tokenizer, self.hparams)
            elif self.hparams.mask_way == 'pmi':
                inputs = batch[0]
                t5_labels = batch[1]
                # inputs, t5_labels = pmi_tokens(batch, self.tokenizer, self.hparams, self.pmi)
            else:
                inputs, labels, t5_labels = mask_tokens(
                    batch, self.tokenizer, self.hparams, self.mask_token)
            loss = self.forward(inputs, t5_labels)

        self.log('test_loss', loss)

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None, ):
        optimizer.step(closure=optimizer_closure)
        # Calculate the new penalty loss for the next update
        if self.hparams.ewc:
            self.ewc_loss = self.hparams.ewc_lambda * self.penalty()

    def configure_optimizers(self):
        if self.hparams.use_adafactor and not self.hparams.adafactor_relative_step:
            adafactor = Adafactor([p for p in self.parameters(
            ) if p.requires_grad], relative_step=self.hparams.adafactor_relative_step,
                                  warmup_init=self.hparams.adafactor_warmup,
                                  scale_parameter=self.hparams.adafactor_scale_params,
                                  lr=self.hparams.lr)

            return [adafactor]
        elif self.hparams.use_adafactor and self.hparams.adafactor_relative_step:
            adafactor = Adafactor([p for p in self.parameters(
            ) if p.requires_grad], relative_step=self.hparams.adafactor_relative_step,
                                  warmup_init=self.hparams.adafactor_warmup,
                                  scale_parameter=self.hparams.adafactor_scale_params)  # lr=self.hparams.lr,

            return [adafactor]
        else:
            adam = AdamW([p for p in self.parameters()
                          if p.requires_grad], lr=self.hparams.lr, eps=1e-08)

            scheduler = get_linear_schedule_with_warmup(
                adam, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)
            self.log('scheduler', scheduler)
            return [adam], [{"scheduler": scheduler, "interval": "step"}]


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        # t = t.cpu()
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model, dataset: list, args, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        self.model.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = True
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.means = {}
        for n, p in deepcopy(self.params).items():
            self.means[n] = variable(p.data)
        self.precision_matrices = self._diag_fisher()
        self.model.cpu()
        torch.cuda.empty_cache()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        print('Caculating fisher information matrix')
        self.model = self.model.cuda()
        for sample in tqdm(self.dataset):
            self.model.zero_grad()
            if self.args.mask_way == 'normal':
                sample = torch.tensor([sample], dtype=torch.long)
                inputs = sample.cuda()
                inputs, _, t5_labels = mask_tokens(inputs, self.tokenizer, self.args, mask_token='<extra_id_0>')
            elif self.args.mask_way == 'ssm' or self.args.mask_way == 'pmi':
                example = [[torch.tensor(sample[0], dtype=torch.long), sample[1], sample[2]]]
                inputs, t5_labels = ssm_tokens(example, self.tokenizer, self.args)
                inputs = inputs.cuda()
                t5_labels = t5_labels.cuda()
            output = self.model(input_ids=inputs, labels=t5_labels)
            loss = output[0]
            loss.backward(create_graph=True)
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss
