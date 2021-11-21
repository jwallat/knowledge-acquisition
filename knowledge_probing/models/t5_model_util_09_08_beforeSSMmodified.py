import sys

import sentry_sdk
import torch
import random
from typing import Any, List, Tuple, Union
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding
from numpy import array, where
from random import choice, randint
from multiprocessing import Manager, Pool, set_start_method
import gc
from time import time


def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if str(type(examples)) != "<class 'torch.Tensor'>":
        examples = torch.tensor(examples)
    # if isinstance(examples[0], (list, tuple)):
    examples = [e.clone().detach() for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result


def tolist(x: Union[List[Any], torch.Tensor]):
    return x.tolist() if isinstance(x, torch.Tensor) else x


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def _whole_word_mask(input_tokens: List[str], max_predictions=512):
    """
    Get 0/1 labels for masked tokens with whole word mask proxy
    """
    # print(input_tokens)
    cand_indexes = []
    for (i, token) in enumerate(input_tokens):

        if len(cand_indexes) >= 1 and not token.startswith("▁"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)

    num_to_predict = min(100, max(1, int(round(len(input_tokens) * 0.15))))  # mlm_probability: float = 0.15
    masked_lms = []
    freeze_indexes = []
    freeze_ids = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        if len(index_set) > 1:
            pp = torch.bernoulli(torch.full((1,), 0.4)).bool()
            if pp:
                for index in index_set:
                    covered_indexes.add(index)
                    masked_lms.append(index)
                p = torch.bernoulli(torch.full((1,), 0.5)).bool()
                if p:
                    freeze_ids.append(index_set)
                    for index in index_set:
                        freeze_indexes.append(index)
                else:
                    pass
            else:
                for index in index_set:
                    ppp = torch.bernoulli(torch.full((1,), 0.5)).bool()
                    if ppp:
                        covered_indexes.add(index)
                        masked_lms.append(index)
        else:
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

    assert len(covered_indexes) == len(masked_lms)

    mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
    return mask_labels, freeze_indexes, freeze_ids


def mask_tokens(examples: torch.Tensor, tokenizer: AutoTokenizer, args, mask_token) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(examples, (dict, BatchEncoding)):
        input_ids = [e["input_ids"] for e in examples]
    else:
        input_ids = examples
        examples = [{"input_ids": e} for e in examples]

    inputs = _collate_batch(input_ids, tokenizer)  # batch_input
    # sys.exit()
    mask_labels = []
    freeze_indexes = []
    freeze_ids = []
    for e in examples:
        ref_tokens = []
        for id in tolist(e["input_ids"]):
            token = tokenizer._convert_id_to_token(id)
            ref_tokens.append(token)
        ml, fr, fi = _whole_word_mask(ref_tokens)
        mask_labels.append(ml)
        freeze_indexes.append(fr)
        freeze_ids.append(fi)

    batch_mask = _collate_batch(mask_labels, tokenizer)

    # mask_tokens
    labels = inputs.clone()
    probability_matrix = batch_mask

    # probability_matrix = torch.tensor(probability_matrix)
    probability_matrix = probability_matrix.to(inputs.device)

    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool)
    special_tokens_mask_tensor = special_tokens_mask_tensor.to(inputs.device)
    probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = probability_matrix.bool()
    ber1 = torch.bernoulli(torch.full(labels.shape, 0.75)).bool().to(inputs.device)
    for i in range(len(ber1)):
        ber1[i][freeze_indexes[i]] = True

    indices_replaced = ber1 & masked_indices
    labels[~indices_replaced] = -100
    # print(indices_replaced[0][11], indices_replaced[0][13], indices_replaced[0][17])
    # inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(mask_token)

    max_seq_length = tokenizer.model_max_length
    t5_labels_tensors = []

    for batch_i, input in enumerate(inputs):
        # For every batch:
        special_tokens_counter = 0
        t5_labels = []
        t5_labels_id = []
        input_clone = input.clone()
        single_batch_indices_replaces = indices_replaced[batch_i]
        freeze_id = freeze_ids[batch_i]

        freeze_idc = freeze_id[:]
        freeze_idd = freeze_indexes[batch_i]

        for i, is_replace in enumerate(single_batch_indices_replaces):
            is_replace = is_replace.item()
            if is_replace is True:
                special_token_id = 32099 - special_tokens_counter
                if i in freeze_idd:
                    for w, ids in enumerate(freeze_idc):
                        if i in ids:
                            t5_labels_id.append(ids)
                            special_tokens_counter = special_tokens_counter + 1
                            input[ids[0]] = special_token_id
                            for ii in ids[1:]:
                                input[ii] = -99
                            del freeze_idc[w]
                else:
                    t5_labels_id.append(i)
                    # special_token_id = 32099 - special_tokens_counter
                    special_tokens_counter = special_tokens_counter + 1
                    # replaced_token = input[i].item()
                    # t5_labels.append(special_token_id)
                    # t5_labels.append(replaced_token)
                    input[i] = special_token_id  # for id_list in freeze_id:
        #     for p in id_list[1:]:
        #         input = del_tensor_ele(input, p)
        idxes = torch.where(input != -99)
        input = input[idxes]
        ex = torch.zeros(len(freeze_idd) - len(freeze_id), device=input.device, dtype=input.dtype)
        inputs[batch_i] = torch.cat((input, ex), 0)

        for i, g in enumerate(t5_labels_id):
            if isinstance(g, List):
                t5_labels.append(32099 - i)
                for x in g:
                    t5_labels.append(input_clone[x].item())
            else:
                t5_labels.append(32099 - i)
                t5_labels.append(input_clone[g].item())

        if len(t5_labels) != 0:
            # Add one more special token as it needs to be one final special token after the last label
            t5_labels.append(32099 - special_tokens_counter)
            t5_labels.append(tokenizer.eos_token_id)
        if len(t5_labels_tensors) == 0:
            # Make sure the first one is of max length so that all will be padded to that length
            num_pad_tokens = max_seq_length - len(t5_labels)
            t5_labels.extend(num_pad_tokens * [tokenizer.pad_token_id])
        t5_labels_tensors.append(torch.tensor(t5_labels, dtype=torch.long))

    # print('Not yet stacked tensors for batches: ', t5_labels_tensors)

    t5_labels_stack = pad_sequence(t5_labels_tensors, batch_first=True)

    t5_labels = t5_labels_stack
    t5_labels = t5_labels.to(inputs.device)

    # 10% of the time, we replace masked input tokens with random word
    ber2 = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device)
    for i in range(len(ber2)):
        ber2[i][freeze_indexes[i]] = False

    indices_random = ber2 & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels, t5_labels


def old_mask_tokens(inputs: torch.Tensor, tokenizer: AutoTokenizer, args) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
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
    special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool)
    special_tokens_mask_tensor = special_tokens_mask_tensor.to(inputs.device)

    probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices = masked_indices.to(inputs.device)

    # make sure that there are no two subsequent masked tokens (t5 requirement)
    for i in range(len(masked_indices[0]) - 2):
        if (masked_indices[0][i] == True) and (masked_indices[0][i + 1] == True):
            # Check if i + 2 is not true, then we can set that to true
            # TODO: Evaluate if droppping is better than shifting
            masked_indices[0][i + 1] = False
            # if masked_indices[0][i+2] != True:
            #     masked_indices[0][i+1] = False
            #     masked_indices[0][i+2] = True
            # else:
            #     # Just drop the second mask
            #     masked_indices[0][i+1] = False

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    full_tensor = torch.full(labels.shape, 0.8)
    full_tensor = full_tensor.to(inputs.device)

    indices_replaced = torch.bernoulli(full_tensor).bool() & masked_indices
    indices_replaced = indices_replaced.to(inputs.device)
    # print('Indices batch to replace: ', indices_replaced)
    labels[~indices_replaced] = -100  # We only compute loss on masked tokens
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

        if len(t5_labels) != 0:
            # Add one more special token as it needs to be one final special token after the last label
            t5_labels.append(32099 - special_tokens_counter)
            t5_labels.append(tokenizer.eos_token_id)

        if len(t5_labels_tensors) == 0:
            # Make sure the first one is of max length so that all will be padded to that length
            num_pad_tokens = max_seq_length - len(t5_labels)
            t5_labels.extend(num_pad_tokens * [tokenizer.pad_token_id])
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


def qa_tokens(inputs: torch.Tensor, tokenizer: AutoTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    input_batch = inputs.clone()
    # print(tokenizer.decode_batch(input_batch))
    question_batch = []
    answer_batch = []
    max_seq_length = tokenizer.model_max_length
    if args.extend4probing:
        ex_q = 200
        ex_a = max_seq_length
    else:
        ex_q = 150
        ex_a = 150
    # print(input_batch[0])
    # print('input_batch', tokenizer.decode(input_batch[0]))
    # sys.exit()
    for input in input_batch:
        # print('error:', tokenizer.decode_batch(input_batch))
        sep_index = input.cpu().numpy().tolist().index(32099)
        question_batch.append(input[:sep_index])
        question_batch[-1] = torch.cat((question_batch[-1], torch.tensor([1], device=input.device, dtype=input.dtype)),
                                       0)
        answer_batch.append(input[sep_index + 1:])
    ex = torch.zeros(ex_q - len(question_batch[0]), device=input.device, dtype=input.dtype)
    question_batch[0] = torch.cat((question_batch[0], ex), 0)
    if not args.extend4probing:
        ex = torch.zeros(ex_a - len(answer_batch[0]), device=input.device, dtype=input.dtype)
        answer_batch[0] = torch.cat((answer_batch[0], ex), 0)
        answer_batch = pad_sequence(answer_batch, batch_first=True)
    question_batch = pad_sequence(question_batch, batch_first=True)

    return question_batch, answer_batch


def ssm_tokens(inputs: torch.Tensor, tokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs_batch = []
    t5_labels_batch = []
    for samp in inputs:
        sent_tokens, entities_sent = samp
        # # print('len', len(tokenizer.tokenize(sent_tokens)))
        # # print('1', entities_sent)
        # if args.ssm_all:
        #     entities_lst = []
        #     for sent_masks in entities_sent:
        #         entities_lst.extend([x for x in sent_masks if sent_masks != []])
        #     mask_num = int(len(tokenizer.tokenize(sent_tokens)) * 0.15)
        #     print('2', entities_lst)
        #     if len(entities_lst) > mask_num:
        #        entities_lst = [entities_lst[i] for i in sorted(random.sample(range(len(entities_lst)), mask_num))]
        #
        #     masked_tokens = []
        #     sents = []
        #     # print('3', entities_lst)
        #     for i, ner in enumerate(entities_lst):
        #         masked_tokens.append('<extra_id_{}> '.format(i) + sent_tokens[ner[2]:ner[3]] + ' ')
        #         if i != (len(entities_lst) - 1):
        #             sents.append(sent_tokens[ner[3]:entities_lst[i + 1][2]])
        #     sents.insert(0, sent_tokens[:entities_lst[0][2]])
        #     sents.append(sent_tokens[entities_lst[-1][3]:])
        #     # print(sents)
        # else:
        #     entities_lst = []
        #     entities_lst.extend([choice(x) for x in entities_sent if x])
        #     masked_tokens = []
        #     sents = []
        #     for i, ner in enumerate(entities_lst):
        #         masked_tokens.append('<extra_id_{}> '.format(i) + sent_tokens[ner[2]:ner[3]] + ' ')
        #         if i != (len(entities_lst) - 1):
        #             sents.append(sent_tokens[ner[3]:entities_lst[i + 1][2]])
        #     sents.insert(0, sent_tokens[:entities_lst[0][2]])
        #     sents.append(sent_tokens[entities_lst[-1][3]:])
        #
        # masked_tokens.append('<extra_id_{}>'.format(len(masked_tokens)))
        # masked_tokens = ''.join(masked_tokens)
        # inputs_sent = []
        # for i, piece in enumerate(sents):
        #     if i < (len(sents) - 1):
        #         inputs_sent.append(piece + '<extra_id_{}>'.format(i))
        #     else:
        #         inputs_sent.append(piece)
        #
        # inputs_sent = ''.join(inputs_sent)
        #sent_tokens = torch.tensor(sent_tokens, dtype=torch.long, device=args.device)
        #entities_sent = torch.tensor(entities_sent, dtype=torch.long, device=args.device)
        # sent_tokens = torch.tensor(sent_tokens, dtype=torch.long, device=args.device)
        # entities_sent = torch.tensor(entities_sent, dtype=torch.long, device=args.device)
        inputs_batch.append(sent_tokens)  # inputs_sent
        t5_labels_batch.append(entities_sent)  # masked_tokens

        ## ssm_tokens end ###
    # try:
    #     inputs_ids_batch = tokenizer(inputs_batch, padding='max_length', return_tensors='pt').input_ids
    # except:
    #     for i in inputs_batch:
    #         print(len(i))
    #         pp = tokenizer(i, padding='max_length', return_tensors='pt').input_ids
    #         print(len(pp[0]))
    #         print(tokenizer.batch_decode(pp))
    #     sys.exit()

    # t5_labels_ids_batch = tokenizer(t5_labels_batch, padding='max_length', return_tensors='pt').input_ids
    # inputs_batch = torch.tensor(inputs_batch, dtype=torch.long, device=args.device)
    # t5_labels_batch = torch.tensor(t5_labels_batch, dtype=torch.long, device=args.device)
    # inputs_ids_batch = inputs_batch.to(args.device)#inputs_ids_batch
    # t5_labels_ids_batch = t5_labels_batch.to(args.device)#t5_labels_ids_batch
    inputs_ids_batch = pad_sequence(inputs_batch, batch_first=True)
    t5_labels_ids_batch = pad_sequence(t5_labels_batch, batch_first=True)
    #print(type(inputs_ids_batch))
    #print(inputs_ids_batch)
    #sys.exit()

    return inputs_ids_batch, t5_labels_ids_batch


def pmi_tokens(inputs: torch.Tensor, tokenizer: AutoTokenizer, args, pmi) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs_batch = inputs.clone()
    inputs_batch = inputs_batch.cpu().numpy().tolist()
    input_batch = []
    t5_labels_batch = []
    for ids in inputs_batch:
        # print(tokenizer.decode(ids))
        ids = list(filter(lambda x: x != tokenizer.pad_token_id, ids))
        ids_str = ' '.join([str(x) for x in ids])
        # print(ids_str)
        ngram_lst = []
        for x in pmi:
            idx = -1
            while True:
                idx = ids_str.find(x, idx + 1)
                if idx == 0:
                    if ids_str[idx + len(x)] == ' ':
                        ngram_lst.append([idx, idx + len(x)])
                elif idx == -1:
                    break
                elif idx + len(x) < len(ids_str) - 1:
                    if ids_str[idx - 1] == ' ' and ids_str[idx + len(x)] == ' ':
                        ngram_lst.append([idx, idx + len(x)])
                elif idx + len(x) == len(ids_str) - 1:
                    if ids_str[idx - 1] == ' ':
                        ngram_lst.append([idx, len(ids_str)])
        # print(ngram_lst)
        # print(len(ngram_lst))
        pmi_ngram = sorted(ngram_lst, key=lambda x: (x[0], x[-1]))
        # print(pmi_ngram)
        # print(len(pmi_ngram))
        i = 0
        sorted_ngram = []
        # print('1', pmi_ngram)
        if len(pmi_ngram) < 2:
            sorted_ngram = pmi_ngram
        else:
            while i < len(pmi_ngram):
                last_term = []
                if not last_term:
                    if i == len(pmi_ngram) - 1:
                        last_term = pmi_ngram[-1]
                    elif pmi_ngram[i][-1] <= pmi_ngram[i + 1][0]:
                        sorted_ngram.append(pmi_ngram[i])
                    elif pmi_ngram[i][0] == pmi_ngram[i + 1][0]:
                        last_term = pmi_ngram[i + 1]
                    elif pmi_ngram[i][-1] >= pmi_ngram[i + 1][-1]:
                        last_term = pmi_ngram[i]
                    elif pmi_ngram[i][-1] < pmi_ngram[i + 1][-1]:
                        last_term = choice(pmi_ngram[i:i + 2])
                    i += 1
                else:
                    if last_term[-1] <= pmi_ngram[i][0]:
                        sorted_ngram.append(last_term)
                        last_term = pmi_ngram[i]
                    elif last_term[0] == pmi_ngram[i][0]:
                        last_term = pmi_ngram[i]
                    elif last_term[-1] >= pmi_ngram[i][-1]:
                        pass
                    elif last_term[-1] < pmi_ngram[i][-1]:
                        last_term = choice([last_term, pmi_ngram[i]])
                    i += 1
                if i == len(pmi_ngram):
                    sorted_ngram.append(last_term)
        # print(sorted_ngram)
        # print(len(sorted_ngram))
        masked_ngram = []
        random.shuffle(sorted_ngram)
        # print(sorted_ngram)
        num_to_predict = int(round(len(ids) * 0.15 * 0.8))
        masked_token_count = 0
        for i in sorted_ngram:
            masked_ngram.append([i[0], i[-1], 1])
            tks = len(ids_str[i[0]:i[-1]].split())
            if masked_token_count + tks > num_to_predict:
                break
            else:
                masked_token_count += tks
        masked_ngram = sorted(masked_ngram, key=lambda x: x[0])
        # print(masked_ngram)
        masked_str_idx = []
        for i in masked_ngram:
            for k in range(i[0], i[1]):
                masked_str_idx.append(k)

        masked_idx_str = ' '.join([str(x) for x in masked_str_idx])
        num_to_replace = int(round(len(ids) * 0.15 * 0.1))
        replace_token_count = 0
        # if masked_token_count < num_to_predict:
        random_idx = [x for x in range(len(ids))]
        random.shuffle(random_idx)
        for idx in random_idx:
            offset = 0
            for token in ids[:idx]:
                offset += len(str(token)) + 1
            random_str = ' '.join([str(x) for x in range(offset, offset + len(str(ids[idx])))])
            if random_str not in masked_idx_str:
                if masked_token_count != num_to_predict:
                    masked_ngram.append([offset, offset + len(str(ids[idx])), 1])
                    masked_token_count += 1
                elif replace_token_count != num_to_replace:
                    masked_ngram.append([offset, offset + len(str(ids[idx])), 0])
                    replace_token_count += 1
                else:
                    break
        masked_ngram = sorted(masked_ngram, key=lambda x: x[0])
        # here test if the process is wrong
        # print('all segmentation included masked and replaced tokens \n', masked_ngram)
        check_str = [ids_str[x[0]:x[1]] for x in masked_ngram]
        idx = [(len(ids_str[:x[0]].split()), len(ids_str[x[0]:x[1]].split())) for x in masked_ngram]
        check_lst = [ids[x[0]:(x[0] + x[-1])] for x in idx]
        if len(' '.join(check_str).split()) != len([e for x in check_lst for e in x]):
            print('PMI-mask process was wrong')
            sys.exit()
        # above test if the process is wrong
        masked_ids = []
        sents = []
        masked_idx = 32099
        if masked_ngram[0][-1]:
            sents.append(ids_str[:masked_ngram[0][0]] + '32099')
            masked_idx -= 1
        else:
            sents.append(ids_str[:masked_ngram[0][0]] + '{}'.format(randint(2, 31199)))
        masked_num = 32099
        for i, ner in enumerate(masked_ngram):
            if ner[-1]:
                masked_ids.append('{} '.format(masked_num) + ids_str[ner[0]:ner[1]])
                masked_num -= 1
            if i != (len(masked_ngram) - 1):
                if masked_ngram[i + 1][-1]:
                    sents.append(ids_str[ner[1] + 1:masked_ngram[i + 1][0]] + '{}'.format(masked_idx))
                    masked_idx -= 1
                else:
                    sents.append(ids_str[ner[1] + 1:masked_ngram[i + 1][0]] + '{}'.format(randint(2, 31199)))
        sents.append(ids_str[masked_ngram[-1][1] + 1:])
        masked_ids.append('{}'.format(masked_num))
        # print('sents', sents)
        # print('masked_ids', masked_ids)
        masked_ids = ' '.join(masked_ids)
        sents = ' '.join(sents)
        sents = [int(x) for x in sents.split()]
        masked_ids = [int(x) for x in masked_ids.split()]
        masked_ids.append(1)
        # print('sents', sents)
        # print('masked_ids', masked_ids)
        # print(tokenizer.decode(sents))
        # print(tokenizer.decode(masked_ids))
        # sys.exit()
        sents = torch.tensor(sents, dtype=inputs.dtype, device=inputs.device)
        masked_ids = torch.tensor(masked_ids, dtype=inputs.dtype, device=inputs.device)
        input_batch.append(sents)
        t5_labels_batch.append(masked_ids)
    ex = torch.zeros(tokenizer.model_max_length - len(input_batch[0]), device=inputs.device, dtype=inputs.dtype)
    input_batch[0] = torch.cat((input_batch[0], ex), 0)
    ex = torch.zeros(max([len(x) for x in t5_labels_batch]) - len(t5_labels_batch[0]), device=inputs.device,
                     dtype=inputs.dtype)
    t5_labels_batch[0] = torch.cat((t5_labels_batch[0], ex), 0)
    input_batch = pad_sequence(input_batch, batch_first=True)
    t5_labels_batch = pad_sequence(t5_labels_batch, batch_first=True)
    return input_batch, t5_labels_batch
