from torch.utils import data
from knowledge_probing.probing.probing import get_probing_dataset_args
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from knowledge_probing.datasets.cloze_data_utils import lowercase_samples, filter_samples, parse_template
from knowledge_probing.file_utils import load_file
from knowledge_probing.probing.probing import get_probing_dataset_args
from dotmap import DotMap

# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)


class TrainClozeDataset(Dataset):
    """
    This is an adoption of the regular cloze dataset that is being used to estimate the amount of knowledge that we can store 
    in models before forgetting happens.

    It is supposed to have different modes:
    1. Using templates as text
    2. Using the evidence sentences

    It also should be able to apply different masking strategies:
    1. Masking the object
    2. Masking a random word in the sequence
    """

    def __init__(self, probing_model: BaseDecoder, tokenizer: PreTrainedTokenizer, args, output_debug_info=False):
        # if not os.path.isfile(args.relation_args.dataset_filename):
        #     print("Could not create features from dataset %s. File not found",
        #           args.relation_args.dataset_filename)
        #     return
        # assert os.path.isfile(args.relation_args.dataset_filename)
        # print("Creating features from dataset file at %s",
        #       args.relation_args.dataset_filename) if output_debug_info else None

        dataset_args = get_probing_dataset_args(args)

        all_elements = []

        for ele in dataset_args:
            dataset_name, relation_args_list = ele

            if 'Google_RE' in dataset_name:
                all_elements.extend(self.load_google_re(
                    args, relation_args_list, probing_model, tokenizer, None))
            elif 'TREx' in dataset_name:
                all_elements.extend(self.load_trex(
                    args, relation_args_list, probing_model, tokenizer, None))
            elif 'ConceptNet' in dataset_name:
                all_elements.extend(self.load_conceptnet(
                    args, relation_args_list, probing_model, tokenizer, None))
            elif 'Squad' in dataset_name:
                all_elements.extend(self.load_squad(
                    args, relation_args_list, probing_model, tokenizer, None))

        # Encode sentences and object labels
        print('Loading the probing datasets')

        for elem in all_elements:

            if 'template' in args.capacity_text_mode:
                # if 'object' in args.capacity_masking_mode:
                elem['sequence'] = tokenizer.encode_plus(elem['sentences'], add_special_tokens=True, return_tensors='pt')[
                    'input_ids'][0]
                # elif 'random' in args.capacity_masking_mode:

            elif 'evidence' in args.capacity_text_mode:
                elem['sequence'] = tokenizer.encode_plus(elem['evidence'], add_special_tokens=True, return_tensors='pt')[
                    'input_ids'][0]

            else:
                raise Exception(
                    'Something wrong with capacity text mode: ', args.capactiy_text_mode)

        self.examples = all_elements
        # elem['sentences'] = tokenizer.encode_plus(elem['sentences'], add_special_tokens=True, return_tensors='pt')[
        #     'input_ids'][0]
        # elem['masked_sentences'] = tokenizer.encode_plus(elem['masked_sentences'], add_special_tokens=True, return_tensors='pt')[
        #     'input_ids'][0]
        # elem['evidence'] = tokenizer.encode_plus(elem['evidence'], add_special_tokens=True, return_tensors='pt')[
        #     'input_ids'][0]
        # elem['masked_evidence'] = tokenizer.encode_plus(elem['masked_evidence'], add_special_tokens=True, return_tensors='pt')[
        #     'input_ids'][0]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def load_google_re(self, args, relation_args_list, probing_model, tokenizer, vocab):
        all_google_samples = []

        for relation_args in relation_args_list:
            relation_args = DotMap(relation_args)
            args.relation_args = relation_args
            samples = load_file(relation_args.dataset_filename)

            if args.lowercase:
                samples = lowercase_samples(samples, probing_model.mask_token)

            samples, _ = filter_samples(
                samples, tokenizer, vocab, relation_args.template)

            # build items with evidence sentence and template sentence
            if relation_args.template and relation_args.template != "":

                for sample in samples:
                    smp = {}
                    smp['sub_label'] = sample['sub_label']
                    smp['obj_label'] = sample['obj_label']
                    smp["sentences"] = parse_template(
                        args.relation_args.template.strip(
                        ), sample["sub_label"].strip(), sample['obj_label']
                    )[0]
                    smp["masked_sentences"] = parse_template(
                        args.relation_args.template.strip(
                        ), sample["sub_label"].strip(), probing_model.mask_token
                    )[0]
                    smp['evidence'] = sample['evidences'][0]['considered_sentences'][0]
                    masked_evidence = sample['masked_sentences'][0]
                    masked_evidence = masked_evidence.replace(
                        '[MASK]', probing_model.mask_token)
                    smp['masked_evidence'] = masked_evidence

                    all_google_samples.append(smp)

        return all_google_samples

    def load_trex(self, args, relation_args_list, probing_model, tokenizer, vocab):
        all_trex_samples = []

        for relation_args in relation_args_list:
            relation_args = DotMap(relation_args)
            args.relation_args = relation_args
            samples = load_file(relation_args.dataset_filename)

            if args.lowercase:
                samples = lowercase_samples(samples, probing_model.mask_token)

            samples, _ = filter_samples(
                samples, tokenizer, vocab, relation_args.template)

            # build items with evidence sentence and template sentence
            if relation_args.template and relation_args.template != "":

                for sample in samples:
                    smp = {}
                    smp['sub_label'] = sample['sub_label']
                    smp['obj_label'] = sample['obj_label']
                    smp["sentences"] = parse_template(
                        args.relation_args.template.strip(
                        ), sample["sub_label"].strip(), sample['obj_label']
                    )[0]
                    smp["masked_sentences"] = parse_template(
                        args.relation_args.template.strip(
                        ), sample["sub_label"].strip(), probing_model.mask_token
                    )[0]
                    evidence = sample['evidences'][0]['masked_sentence'][0]
                    smp['evidence'] = evidence.replace(
                        '[MASK]', sample['obj_label'])
                    masked_evidence = evidence.replace(
                        '[MASK]', probing_model.mask_token)
                    smp['masked_evidence'] = masked_evidence

                    all_trex_samples.append(smp)

        return all_trex_samples

    def load_conceptnet(self, args, relation_args_list, probing_model, tokenizer, vocab):
        all_conceptnet_samples = []

        for relation_args in relation_args_list:
            relation_args = DotMap(relation_args)
            args.relation_args = relation_args
            samples = load_file(relation_args.dataset_filename)

            if args.lowercase:
                samples = lowercase_samples(samples, probing_model.mask_token)

            samples, _ = filter_samples(
                samples, tokenizer, vocab, relation_args.template)

            # build items with evidence sentence and template sentence

            for sample in samples:
                smp = {}
                smp['sub_label'] = sample['sub_label']
                smp['obj_label'] = sample['obj_label']
                masked_sentences = sample['masked_sentences'][0]
                smp["sentences"] = masked_sentences.replace(
                    '[MASK]', sample['obj_label'])
                smp["masked_sentences"] = masked_sentences.replace(
                    '[MASK]', probing_model.mask_token)

                smp['evidence'] = smp["sentences"]
                smp['masked_evidence'] = smp["masked_sentences"]

                all_conceptnet_samples.append(smp)

        return all_conceptnet_samples

    def load_squad(self, args, relation_args_list, probing_model, tokenizer, vocab):
        all_squad_samples = []

        for relation_args in relation_args_list:
            relation_args = DotMap(relation_args)
            args.relation_args = relation_args
            samples = load_file(relation_args.dataset_filename)

            if args.lowercase:
                samples = lowercase_samples(samples, probing_model.mask_token)

            samples, _ = filter_samples(
                samples, tokenizer, vocab, relation_args.template)

            # build items with evidence sentence and template sentence

            for sample in samples:
                smp = {}
                smp['sub_label'] = sample['sub_label']
                smp['obj_label'] = sample['obj_label']
                masked_sentences = sample['masked_sentences'][0]
                smp["sentences"] = masked_sentences.replace(
                    '[MASK]', sample['obj_label'])
                smp["masked_sentences"] = masked_sentences.replace(
                    '[MASK]', probing_model.mask_token)

                smp['evidence'] = smp["sentences"]
                smp['masked_evidence'] = smp["masked_sentences"]

                all_squad_samples.append(smp)

        return all_squad_samples
