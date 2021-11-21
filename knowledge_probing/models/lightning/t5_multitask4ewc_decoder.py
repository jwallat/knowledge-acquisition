from knowledge_probing.models.lightning.t5_decoder import T5Decoder
import sys
import torch
from transformers.optimization import Adafactor
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from knowledge_probing.probing.metrics import calculate_metrics, calculate_metrics4paq
from knowledge_probing.datasets.text_dataset import TextDataset
from knowledge_probing.models.t5_model_util import mask_tokens, qa_tokens, ssm_tokens, pmi_tokens
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import random
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable
import pickle5 as pickle


class T5MultitaskDecoder(T5Decoder):

    def __init__(self, hparams):
        super(T5MultitaskDecoder, self).__init__(hparams)

        self.save_hyperparameters(hparams)  # self.hparams = hparams
        self.model, self.config = self.prepare_model(
            hparams=self.hparams)  # , self.decoder
        if self.hparams.ewc:
            self.ewc = 0
        self.backup_data = []

    def prepare_ewc(self, dataset=None):
        old_train_dataset = dataset
        print(len(old_train_dataset), type(old_train_dataset))
        ewc = EWC(self.model, old_train_dataset, self.hparams, self.tokenizer)
        print('Prepare EWC ready!')
        return ewc

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        outputs = self.model(input_ids=inputs, labels=labels)
        loss = outputs[0]
        if self.hparams.ewc and self.global_step > 0:
            loss = loss + self.hparams.ewc_lambda * self.ewc.penalty(self.model)
        return loss

    def training_step(self, batch, batch_idx):
        self.backup_data.extend(list(batch[-1].clone().cpu()))
        if len(self.backup_data) == 4:
            self.ewc = self.prepare_ewc(self.backup_data)
        if self.hparams.finetuning:
            inputs, t5_labels = qa_tokens(batch, self.tokenizer, self.hparams)
        elif self.hparams.multitask:
            if batch[0] == 'QA':
                inputs, t5_labels = qa_tokens(batch[-1], self.tokenizer, self.hparams)
            if batch[0] == 'text':
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

    def optimizer_step(self,
                       epoch: int = None,
                       batch_idx: int = None,
                       optimizer: Optimizer = None,
                       optimizer_idx: int = None,
                       optimizer_closure: Optional[Callable] = None,
                       on_tpu: bool = None,
                       using_native_amp: bool = None,
                       using_lbfgs: bool = None, ):
        optimizer.step(closure=optimizer_closure)
        print(len(self.backup_data))
        self.ewc = self.prepare_ewc(dataset=self.backup_data)


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda(1)
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model, dataset: list, args, tokenizer):
        self.model = model.cuda(1)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        self.model.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = True
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        print('Caculating fisher information matrix')
        for sample in tqdm(self.dataset):
            self.model.zero_grad()
            if self.args.mask_way == 'normal':
                sample = torch.tensor([sample], dtype=torch.long)
                inputs = variable(sample)
                inputs, _, t5_labels = mask_tokens(inputs, self.tokenizer, self.args, mask_token='<extra_id_0>')
            output = self.model(input_ids=inputs, labels=t5_labels)
            loss = output[0]
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
