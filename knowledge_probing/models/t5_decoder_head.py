from torch import nn
from torch.nn import CrossEntropyLoss
import torch


class T5DecoderHead(nn.Module):

    def __init__(self, config, pre_trained_lm_layer=None) -> None:
        super().__init__()
        self.config = config
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

        if pre_trained_lm_layer:
            self.lm_head = pre_trained_lm_layer

    def forward(self, hidden_states, labels):
        # Apply layernorm and dropout as it is done in the original T5 (T5Stack)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        prediction_scores = self.lm_head(hidden_states)  # Aka lm_logits
        outputs = (prediction_scores,)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prediction_scores.view(-1,
                                                   prediction_scores.size(-1)), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states
