import torch
from typing import Tuple, List
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


def mask_tokens(inputs: torch.Tensor, tokenizer: AutoTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
        This is an adapted version to the regular mask_tokens function for T5. It will also produce the T5-labels that are needed
        fo the mlm-training. 
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. 
        This is the standard script used in the huggingface libaray with slight adjustments for pytorch-lightning. 
        That is only adjusting how tensors are casted to the device (e.g. probability_matrix = probability_matrix.to(inputs.device)).
    """

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    probability_matrix = probability_matrix.to(inputs.device)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask_tensor = torch.tensor(
        special_tokens_mask, dtype=torch.bool)
    special_tokens_mask_tensor = special_tokens_mask_tensor.to(inputs.device)

    probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices = masked_indices.to(inputs.device)

    # make sure that there are no two subsequent masked tokens (t5 requirement)
    for i in range(len(masked_indices[0]) - 2):
        if (masked_indices[0][i] == True) and (masked_indices[0][i+1] == True):
            # Check if i + 2 is not true, then we can set that to true
            # TODO: Evaluate if droppping is better than shifting
            masked_indices[0][i+1] = False
            # if masked_indices[0][i+2] != True:
            #     masked_indices[0][i+1] = False
            #     masked_indices[0][i+2] = True
            # else:
            #     # Just drop the second mask
            #     masked_indices[0][i+1] = False

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    full_tensor = torch.full(labels.shape, 0.8)
    full_tensor = full_tensor.to(inputs.device)

    indices_replaced = torch.bernoulli(full_tensor).bool() & masked_indices
    indices_replaced = indices_replaced.to(inputs.device)
    # print('Indices batch to replace: ', indices_replaced)

    max_seq_length = tokenizer.model_max_length
    t5_labels_tensors = []

    for i, input in enumerate(inputs):
        # For every batch:
        special_tokens_counter = 0
        t5_labels = []
        single_batch_indices_replaces = indices_replaced[i]
        for i, is_replace in enumerate(single_batch_indices_replaces):
            is_replace = is_replace.item()
            if is_replace is True:
                special_token_id = 32099 - special_tokens_counter
                special_tokens_counter = special_tokens_counter + 1
                replaced_token = input[i].item()
                # print('Replaced_token: ', replaced_token)
                t5_labels.append(special_token_id)
                t5_labels.append(replaced_token)
                input[i] = special_token_id

        if len(t5_labels) is not 0:
            # Add one more special token as it needs to be one final special token after the last label
            t5_labels.append(32099 - special_tokens_counter)
            t5_labels.append(tokenizer.eos_token_id)

        if len(t5_labels_tensors) is 0:
            # Make sure the first one is of max length so that all will be padded to that length
            num_pad_tokens = max_seq_length - len(t5_labels)
            t5_labels.extend(num_pad_tokens*[tokenizer.pad_token_id])
        t5_labels_tensors.append(torch.tensor(t5_labels, dtype=torch.long))
    # print('Not yet stacked tensors for batches: ', t5_labels_tensors)

    t5_labels_stack = pad_sequence(t5_labels_tensors, batch_first=True)

    t5_labels = t5_labels_stack
    t5_labels = t5_labels.to(inputs.device)

    # 10% of the time, we replace masked input tokens with random word
    other_full_tensor = torch.full(labels.shape, 0.5)
    other_full_tensor = other_full_tensor.to(inputs.device)

    indices_random = torch.bernoulli(
        other_full_tensor).bool() & masked_indices & ~indices_replaced
    indices_random = indices_random.to(inputs.device)

    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    random_words = random_words.to(inputs.device)

    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, t5_labels
