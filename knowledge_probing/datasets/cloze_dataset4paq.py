import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os, sys
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from knowledge_probing.datasets.cloze_data_utils4paq import lowercase_samples4paq, pad_or_truncate, filter_samples, parse_template
from knowledge_probing.file_utils import load_file
import pandas as pd
import pickle
# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)
# used in probing.probe


class ClozeDataset4paq(Dataset):
    def __init__(self, probing_model: BaseDecoder, tokenizer: PreTrainedTokenizer, args, block_size=512, output_debug_info=True):
        if not os.path.isfile(args.paq_probing_path):
            print("Could not create features from dataset %s. File not found",
                  args.paq_probing_path)
            return
        assert os.path.isfile(args.paq_probing_path)
        print("Creating features from dataset file at %s",
              args.paq_probing_path) if output_debug_info else None

        self.samples = []

        samples = load_file(args.paq_probing_path)

        print('number samples: {}'.format(len(samples))
              ) if output_debug_info else None

        encoded_samples = []
        i = 0
        for sample in samples:
            if args.lowercase:
                print("lowercasing all samples...") if output_debug_info else None
                sample = lowercase_samples4paq(sample)
            sample["uuid"] = i
            i += 1
            loss_label = tokenizer(sample['answer'])['input_ids']
            loss_label.pop()
            sample['len'] = len(loss_label)
            if not probing_model.probe_encoder:
                sample['inputs_tokens'] = sample['question'].replace(sample['Interrogative_word']
                                                                     , probing_model.mask_token)
            else:
                label_mask_tokens = ''
                for i in range(len(loss_label)):
                    if i == (len(loss_label)-1):
                        label_mask_tokens = label_mask_tokens + '<extra_id_'+str(i)+'>'
                    else:
                        label_mask_tokens = label_mask_tokens + '<extra_id_'+str(i)+'> '
                    sample['inputs_tokens'] = sample['question'].replace(sample['Interrogative_word']
                                                                         , label_mask_tokens)
            sample['inputs_id'] = torch.tensor(tokenizer(sample['inputs_tokens'],
                                            padding='max_length').input_ids)

            no_query = sample['question'].replace(sample['Interrogative_word'], '')
            no_query = tokenizer(no_query).input_ids
            for i in range(len(no_query)):
                no_query[i] = -100
            loss_label.extend(no_query)
            loss_label = pad_or_truncate(loss_label, 512)
            sample['labels'] = torch.tensor(loss_label)
            t5_labels = tokenizer('<extra_id_0> ' + sample['answer'] + ' <extra_id_1>').input_ids
            t5_labels = pad_or_truncate(t5_labels, 30)
            sample['t5_labels'] = t5_labels
            encoded_samples.append(sample)


        self.samples = encoded_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]
