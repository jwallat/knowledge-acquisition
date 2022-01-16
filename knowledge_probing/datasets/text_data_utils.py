import torch
from typing import Tuple, List
from transformers import AutoTokenizer
import os, sys
from random import choice
import pickle5 as pickle
from knowledge_probing.datasets.text_dataset import TextDataset
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_in_chunks(filePath, chunk_size=1024 * 8):
    """
    me: Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1M
    You can set your own chunk size
    """
    file_object = open(filePath, 'r', encoding='utf-8')
    while True:
        chunk_data = file_object.read(chunk_size)
        if not chunk_data:
            break
        yield chunk_data


def mask_tokens(inputs: torch.Tensor, tokenizer: AutoTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. 
        This is the standard script used in the huggingface libaray with slight adjustments for pytorch-lightning. 
        That is only adjusting how tensors are casted to the device (e.g. probability_matrix = probability_matrix.to(inputs.device)).
    """

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
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

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    full_tensor = torch.full(labels.shape, 0.8)
    full_tensor = full_tensor.to(inputs.device)

    indices_replaced = torch.bernoulli(full_tensor).bool() & masked_indices
    indices_replaced = indices_replaced.to(inputs.device)

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

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
    return inputs, labels


def datasets_handle(file_path, args, tokenizer, block_size=512):
    print(file_path)
    assert os.path.isfile(file_path)
    block_size = block_size - \
                 (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
    directory, filename = os.path.split(file_path)
    model_type_string = args.model_type.replace('/', '-')
    if args.mask_way == 'ssm' and 'jsonl' not in file_path:
        cached_features_file = os.path.join(directory, model_type_string + "_cached_SSM_" + filename)
    elif args.mask_way == 'pmi' and 'jsonl' not in file_path:
        cached_features_file = os.path.join(directory, model_type_string +
                                            "_cached_PMI_" + str(block_size) + "_" + filename)
    elif args.mask_way == 'normal' or 'jsonl' in file_path:
        cached_features_file = os.path.join(directory,
                                            model_type_string + "_cached_lm_" + str(block_size) + "_" + filename)
    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)

    else:
        TextDataset(tokenizer, args, file_path=file_path,
                    block_size=tokenizer.model_max_length)
    assert os.path.isfile(cached_features_file)
    with open(cached_features_file, "rb") as handle:
        examples = pickle.load(handle)

    return examples


def ssm_tokens(inputs, tokenizer, args):
    sent_tokens, entities_sent = inputs
    # print('len', len(tokenizer.tokenize(sent_tokens)))
    # print('1', entities_sent)
    if args.ssm_all:
        entities_lst = []
        for sent_masks in entities_sent:
            entities_lst.extend([x for x in sent_masks if sent_masks != []])
        # mask_num = int(len(tokenizer.tokenize(sent_tokens)) * 0.15)
        # print('2', entities_lst)
        # if len(entities_lst) > mask_num:
        #    entities_lst = [entities_lst[i] for i in sorted(random.sample(range(len(entities_lst)), mask_num))]

        masked_tokens = []
        sents = []
        # print('3', entities_lst)
        for i, ner in enumerate(entities_lst):
            masked_tokens.append('<extra_id_{}> '.format(i) + sent_tokens[ner[2]:ner[3]] + ' ')
            if i != (len(entities_lst) - 1):
                sents.append(sent_tokens[ner[3]:entities_lst[i + 1][2]])
        sents.insert(0, sent_tokens[:entities_lst[0][2]])
        sents.append(sent_tokens[entities_lst[-1][3]:])
        # print(sents)
    else:
        entities_lst = []
        entities_lst.extend([choice(x) for x in entities_sent if x])
        masked_tokens = []
        sents = []
        for i, ner in enumerate(entities_lst):
            masked_tokens.append('<extra_id_{}> '.format(i) + sent_tokens[ner[2]:ner[3]] + ' ')
            if i != (len(entities_lst) - 1):
                sents.append(sent_tokens[ner[3]:entities_lst[i + 1][2]])
        sents.insert(0, sent_tokens[:entities_lst[0][2]])
        sents.append(sent_tokens[entities_lst[-1][3]:])

    masked_tokens.append('<extra_id_{}>'.format(len(masked_tokens)))
    masked_tokens = ''.join(masked_tokens)
    inputs_sent = []
    for i, piece in enumerate(sents):
        if i < (len(sents) - 1):
            inputs_sent.append(piece + '<extra_id_{}>'.format(i))
        else:
            inputs_sent.append(piece)

    inputs_sent = ''.join(inputs_sent)

    ## ssm_tokens end ###
    inputs_ids = tokenizer(inputs_sent, padding='max_length').input_ids  # return_tensors='pt'
    t5_labels_ids = tokenizer(masked_tokens, padding='max_length').input_ids  # , return_tensors='pt'

    return inputs_ids, t5_labels_ids


def eliminate_underline(lst=None, item=3):
    if lst[0] == 3:
        lst.pop(0)
    return [value for (index, value) in enumerate(lst) if
            (value == item and lst[index - 1] < 32000) or value != item or index == 0]


#Similar to SSM processing. Search the included PMI vocabulary for each text, then mark span.
def pmi_tokens(inputs, pmi_vocab, pad_token_id):
    inputs = list(filter(lambda x: x != pad_token_id, inputs))
    ids_str = ' '.join([str(x) for x in inputs])
    ngram_lst = []
    for x in pmi_vocab:
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
    pmi_ngram = sorted(ngram_lst, key=lambda x: (x[0], x[-1]))
    i = 0
    sorted_ngram = []
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
    i = 0
    pmis_idx = []
    offset = 0
    remapping = []
    for id in inputs[:-1]:
        len_of_id = len(str(id))
        remapping.append([offset, offset + len_of_id])
        offset += len_of_id + 1

    for pmi_idx in sorted_ngram:
        continuous = False
        while i < len(inputs) - 1:
            if remapping[i][0] == pmi_idx[0] and remapping[i][1] == pmi_idx[-1]:
                pmis_idx.append([i])
                if i + 1 != (len(inputs) - 1):
                    if remapping[i] == remapping[i + 1]:
                        pmis_idx.append([i + 1])
                        i += 2
                        break
                i += 1
                break
            elif remapping[i][0] == pmi_idx[0] and remapping[i][1] != pmi_idx[-1]:
                pmis_idx.append([i])
                continuous = True
            elif continuous and remapping[i][1] < pmi_idx[-1]:
                pmis_idx[-1].append(i)
            elif continuous and remapping[i][1] == pmi_idx[-1]:
                pmis_idx[-1].append(i)
                if i + 1 != (len(inputs) - 1):
                    if remapping[i] == remapping[i + 1]:
                        pmis_idx[-1].append(i + 1)
                        i += 2
                        break
                i += 1
                break
            i += 1
    en_ids = [k for x in pmis_idx for k in x]
    others_idx = list(set(range(len(inputs[:-1]))).difference(set(en_ids)))
    return pmis_idx, others_idx
