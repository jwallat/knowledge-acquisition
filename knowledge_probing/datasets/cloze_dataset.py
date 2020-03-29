from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import pickle
import json
import torch
from knowledge_probing.datasets.cloze_data_utils import lowercase_samples, filter_samples, get_index_for_mask, parse_template
from knowledge_probing.file_utils import load_file


class ClozeDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, vocab, block_size=512, output_debug_info=False):
        if not os.path.isfile(args.probe.relation_args.dataset_filename):
            print("Could not create features from dataset %s. File not found",
                  args.probe.relation_args.dataset_filename)
            return
        assert os.path.isfile(args.probe.relation_args.dataset_filename)
        print("Creating features from dataset file at %s",
              args.probe.relation_args.dataset_filename) if output_debug_info else None

        samples = load_file(args.probe.relation_args.dataset_filename)
        print('number samples: {}'.format(len(samples))
              ) if output_debug_info else None

        # self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        # self.samples = samples

        # Lowercase if needed
        if args.lowercase:
            print("lowercasing all samples...") if output_debug_info else None
            samples = lowercase_samples(samples, tokenizer.mask_token)

        # Filter samples
        print('filtering the samples') if output_debug_info else None
        samples, msg = filter_samples(
            samples, tokenizer, vocab, args.probe.relation_args.template)
        print('number filtered samples: {}'.format(
            len(samples))) if output_debug_info else None
        # print(msg)

        # Make sure every sub/obj pair is only used once
        if args.probe.relation_args.template and args.probe.relation_args.template != "":
            facts = []
            for sample in samples:
                # print(sample)
                sub = sample["sub_label"]
                obj = sample["obj_label"]
                if 'judgments' in sample and ((sub, obj) not in facts):
                    facts.append((sub, obj, sample['judgments']))
                elif (sub, obj) not in facts:
                    facts.append((sub, obj))
            print("distinct template facts: {}".format(
                len(facts))) if output_debug_info else None
            all_samples = []
            for fact in facts:
                sample = {}
                if len(fact) == 2:
                    (sub, obj) = fact
                elif len(fact) == 3:
                    (sub, obj, judgments) = fact
                    sample['judgments'] = judgments
                sample["sub_label"] = sub
                sample["obj_label"] = obj

                # substitute all sentences with a standard template
                sample["masked_sentences"] = parse_template(
                    args.probe.relation_args.template.strip(
                    ), sample["sub_label"].strip(), tokenizer.mask_token
                )
                all_samples.append(sample)
            samples = all_samples

        # Give every sample a uuid
        i = 0
        for sample in samples:
            if "uuid" not in sample:
                sample["uuid"] = i
            i += 1

        # print(samples[0])

        # Encode sentences and object label
        # tokenizer.padding_side = 'left'
        encoded_samples = []
        for sample in samples:
            encoded_sample = {}
            # print(sample['masked_sentences'][0])
            encoded_sample['masked_sentences'] = tokenizer.encode_plus(sample['masked_sentences'][0], add_special_tokens=True, return_tensors='pt')[
                'input_ids'][0]  # return_tensors='pt' pad_to_max_length=True,
            # print(encoded_sample)
            # Since we are only funnelling masked_sentences into bert #tokenizer.encode_plus(sample['obj_label'], add_special_tokens=False, pad_to_max_length=True, return_tensors='pt')['input_ids']
            encoded_sample['obj_label'] = sample['obj_label']
            encoded_sample['mask_index'] = get_index_for_mask(
                encoded_sample['masked_sentences'], tokenizer.mask_token_id)
            encoded_sample['uuid'] = sample["uuid"]
            if 'judgments' in sample:
                # print('judgments in sample')
                encoded_sample['judgments'] = sample['judgments']
            encoded_samples.append(encoded_sample)

        self.samples = encoded_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]
