from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from knowledge_probing.datasets.cloze_data_utils import lowercase_samples, filter_samples, parse_template
from knowledge_probing.file_utils import load_file

# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)


class ClozeDataset(Dataset):
    def __init__(self, probing_model: BaseDecoder, tokenizer: PreTrainedTokenizer, args, vocab, block_size=512, output_debug_info=False):
        if not os.path.isfile(args.relation_args.dataset_filename):
            print("Could not create features from dataset %s. File not found",
                  args.relation_args.dataset_filename)
            return
        assert os.path.isfile(args.relation_args.dataset_filename)
        print("Creating features from dataset file at %s",
              args.relation_args.dataset_filename) if output_debug_info else None

        self.samples = []

        samples = load_file(args.relation_args.dataset_filename)
        print('number samples: {}'.format(len(samples))
              ) if output_debug_info else None

        # Lowercase if needed
        if args.lowercase:
            print("lowercasing all samples...") if output_debug_info else None
            samples = lowercase_samples(samples, probing_model.mask_token)

        # Filter samples
        print('filtering the samples') if output_debug_info else None
        samples, _ = filter_samples(
            samples, tokenizer, vocab, args.relation_args.template)
        print('number filtered samples: {}'.format(
            len(samples))) if output_debug_info else None

        # Make sure every sub/obj pair is only used once
        if args.relation_args.template and args.relation_args.template != "":
            facts = []
            for sample in samples:
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
                sub = None
                obj = None
                if len(fact) == 2:
                    (sub, obj) = fact
                elif len(fact) == 3:
                    (sub, obj, judgments) = fact
                    sample['judgments'] = judgments
                sample["sub_label"] = sub
                sample["obj_label"] = obj

                # substitute all sentences with a standard template
                sample["masked_sentences"] = parse_template(
                    args.relation_args.template.strip(
                    ), sample["sub_label"].strip(), probing_model.mask_token
                )
                all_samples.append(sample)
            samples = all_samples

        # Give every sample a uuid
        i = 0
        for sample in samples:
            if "uuid" not in sample:
                sample["uuid"] = i
            i += 1

        # Encode sentences and object label
        encoded_samples = []
        for sample in samples:
            encoded_sample = {}
            # Get the correct mask into the model
            if '[MASK]' in sample['masked_sentences'][0]:
                sample['masked_sentences'][0] = sample['masked_sentences'][0].replace(
                    '[MASK]', probing_model.mask_token)
            encoded_sample['masked_sentences'] = tokenizer.encode_plus(sample['masked_sentences'][0], add_special_tokens=True, return_tensors='pt')[
                'input_ids'][0]
            encoded_sample['obj_label'] = sample['obj_label']

            if 't5' in args.model_type:
                encoded_sample['t5_labels'] = probing_model.get_probing_t5_labels(input_ids_tensor=encoded_sample['masked_sentences'],
                                                                                  obj_label=encoded_sample['obj_label'])
                # print('Un-Masked sentence: ', sample['masked_sentences'][0])
                # print('Masked sentence: ', encoded_sample['masked_sentences'])
                # print('T5 labels: ', encoded_sample['t5_labels'])
                encoded_sample['mask_index'] = probing_model.get_index_for_masked_token(
                    encoded_sample['masked_sentences'], encoded_sample['t5_labels'])
            else:
                encoded_sample['t5_labels'] = None
                encoded_sample['mask_index'] = probing_model.get_index_for_masked_token(
                    encoded_sample['masked_sentences'])

            encoded_sample['uuid'] = sample["uuid"]
            if 'judgments' in sample:
                encoded_sample['judgments'] = sample['judgments']
            encoded_samples.append(encoded_sample)

        self.samples = encoded_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]
