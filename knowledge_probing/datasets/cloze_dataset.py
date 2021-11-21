from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os, sys
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from knowledge_probing.datasets.cloze_data_utils import lowercase_samples, filter_samples, parse_template
from knowledge_probing.file_utils import load_file

# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)

# used in probing.probe
class ClozeDataset(Dataset):
    def __init__(self, probing_model: BaseDecoder, tokenizer: PreTrainedTokenizer, args, vocab, block_size=512, output_debug_info=False):
        if not os.path.isfile(args.relation_args.dataset_filename):
            print("Could not create features from dataset %s. File not found",
                  args.relation_args.dataset_filename)
            return
        assert os.path.isfile(args.relation_args.dataset_filename)
        print("Creating features from dataset file at %s",
              args.relation_args.dataset_filename) if output_debug_info else None       #不打印

        self.samples = []

        samples = load_file(args.relation_args.dataset_filename)
        #print('samples', samples)
        print('number samples: {}'.format(len(samples))
              ) if output_debug_info else None

        # Lowercase if needed
        if args.lowercase:
            print("lowercasing all samples...") if output_debug_info else None
            samples = lowercase_samples(samples, probing_model.mask_token)

        # Filter samples
        print('filtering the samples') if output_debug_info else None
        #print('plate:', args.relation_args.template)
        samples, _ = filter_samples(
            samples, tokenizer, vocab, args.relation_args.template)
        print('number filtered samples: {}'.format(
            len(samples))) if output_debug_info else None
        # Make sure every sub/obj pair is only used once
        if args.relation_args.template and args.relation_args.template != "":
            facts = []
            for sample in samples:
                #print('sample', sample)
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

                # substitute all sentences with a standard template  用标准模板替换所有句子
                sample["masked_sentences"] = parse_template(args.relation_args.template.strip(), sample["sub_label"].strip(), probing_model.mask_token)
                sample['temp_sentences'] = parse_template(args.relation_args.template.strip(), sample["sub_label"].strip(), sample["obj_label"].strip())
                all_samples.append(sample)
            samples = all_samples

        elif args.relation_args.template == "":
            all_samples = []
            for sample in samples:
                samp = {}
                samp['masked_sentences'] = []
                samp['temp_sentences'] = []
                #print('sample!!!: ', sample)
                # substitute all sentences with a standard template  用标准模板替换所有句子
                samp['obj_label'] = sample['obj_label']
                samp['sub_label'] = sample['sub_label']
                obj_tokens = tokenizer.encode_plus(sample['obj_label'], add_special_tokens=False)['input_ids']
                # if len(obj_tokens) > 1:
                #     second_obj_token = tokenizer.decode(obj_tokens[1:])
                #     second_obj_token = second_obj_token.replace('#', '')
                #     samp['masked_sentences'].append(sample['masked_sentences'][0].replace('[MASK]', '[MASK]'+second_obj_token))
                # else:
                samp['masked_sentences'] = sample['masked_sentences']
                if 't5' in args.model_type:
                    samp['temp_sentences'].append(sample['masked_sentences'][0].replace('[mask]', sample['obj_label']))
                else:
                    samp['temp_sentences'].append(sample['masked_sentences'][0].replace('[MASK]', sample['obj_label']))
                all_samples.append(samp)
            samples = all_samples

            #print(samples)
        # Give every sample a uuid 识别码？
        i = 0
        for sample in samples:     # samples 中包含了替换的模板 原来的masked_sentences被覆盖了
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
            elif 't5' in args.model_type and '[mask]' in sample['masked_sentences'][0]:
                sample['masked_sentences'][0] = sample['masked_sentences'][0].replace(
                    '[mask]', probing_model.mask_token)

            encoded_sample['masked_sentences'] = tokenizer.encode_plus(sample['masked_sentences'][0], add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            encoded_sample['temp_sentences'] = tokenizer.encode_plus(sample['temp_sentences'][0], add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            encoded_sample['obj_label'] = sample['obj_label']
            if len(encoded_sample['masked_sentences']) != len(encoded_sample['temp_sentences']):
                mask_id = encoded_sample['masked_sentences'].numpy().tolist().index(probing_model.mask_token_id)
                encoded_sample['masked_sentences'] = encoded_sample['temp_sentences']
                encoded_sample['masked_sentences'][mask_id]= probing_model.mask_token_id

            if 't5' in args.model_type:
                encoded_sample['t5_labels'] = probing_model.get_probing_t5_labels(input_ids_tensor=encoded_sample['masked_sentences'],
                                                                                  obj_label=encoded_sample['obj_label'])
                # print('Un-Masked sentence: ', sample['masked_sentences'][0])

                encoded_sample['mask_index'] = probing_model.get_index_for_masked_token(
                    encoded_sample['masked_sentences'], encoded_sample['t5_labels'])
            else:
                encoded_sample['t5_labels'] = None
                encoded_sample['mask_index'] = probing_model.get_index_for_masked_token(
                    encoded_sample['masked_sentences'])                     #获取[mask]的index
            #print(encoded_sample)
            #print(encoded_sample['masked_sentences'])
            #print(encoded_sample['temp_sentences'])

            encoded_sample['uuid'] = sample["uuid"]
            if 'judgments' in sample:
                encoded_sample['judgments'] = sample['judgments']
            encoded_samples.append(encoded_sample)


        #print(encoded_samples)
        self.samples = encoded_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]
