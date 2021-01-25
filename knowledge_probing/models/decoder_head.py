from transformers.activations import gelu, gelu_new
from torch import nn
from torch.nn import CrossEntropyLoss
import torch


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu,
          "gelu_new": gelu_new, "mish": mish}

LayerNorm = torch.nn.LayerNorm


class DecoderTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # if isinstance(config.hidden_act, str):
        #     self.transform_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.transform_act_fn = config.hidden_act

        # Use BERT default gelu activation and layer norm eps
        self.transform_act_fn = gelu

        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=1e-12)
        # config.hidden_size, eps=1e-6)
        # TODO: Only for test t5

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DecoderPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DecoderTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MyDecoderHead(nn.Module):
    def __init__(self, config, pre_trained_head=None, pre_trained_fc_layer=None):
        super().__init__()
        self.config = config
        self.predictions = DecoderPredictionHead(config)
        self.apply(self._init_weights)

        if pre_trained_head:
            # Replace whole prediction head with a pre-trained one
            self.predictions = pre_trained_head
            # self.predictions.train()
            # self.predictions.requires_grad = True
            # for param in self.predictions.parameters():
            #     param.requires_grad = True

        if pre_trained_fc_layer:
            # Only replace the fc layer
            self.predictions.decoder = pre_trained_fc_layer
            print(self.predictions)
            # self.pre_trained_layer = pre_trained_fc_layer

    def forward(self, sequence_output,
                masked_lm_labels):
        # if self.pre_trained_layer:
        #     prediction_scores = self.pre_trained_layer(sequence_output)
        # else:
        prediction_scores = self.predictions(sequence_output)

        outputs = (prediction_scores,)

        # if masked_lm_labels is not None:
        # -100 index = padding token
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        # masked_lm_loss = loss_fct(
        #     prediction_scores.view(-1, prediction_scores.size(-1)), masked_lm_labels.view(-1))
        outputs = (masked_lm_loss,) + outputs
        # else:
        #     print('No mlm labels in decoding head!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
