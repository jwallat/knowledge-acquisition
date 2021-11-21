import sys
from typing import Tuple
from torch.utils.data import Dataset
import os
import pickle
from tqdm import tqdm
import torch
from knowledge_probing.datasets.text_data_utils import chunks, read_in_chunks, pmi_tokens
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from dotmap import DotMap
import pickle5 as pickle

"""Note that for SSM and PMI, the tokenized tokens sequence and all the specific tokens with their spans included in 
# sequence will be pickled together. Then, when processing data during training, they will be randomly selected to meet 
# the 15% corruption rate."""


class TextDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, args, file_path: str, block_size=512):
        print(file_path)
        self.args = args
        assert os.path.isfile(file_path)

        block_size = block_size - \
                     (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        model_type_string = args.model_type.replace('/', '-')
        # pickling for random tokens masking
        if args.mask_way == 'normal':
            cached_features_file = os.path.join(
                directory, model_type_string +
                           "_cached_lm_" + str(block_size) + "_" + filename)

            if os.path.exists(cached_features_file):
                print("Loading features from cached file %s", cached_features_file)
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                print("Creating features from dataset file at %s", directory)

                tokenizer.save_vocabulary(save_directory='.')

                print('---' * 10)
                print("Saving features into cached file %s", cached_features_file)
                filesize = os.path.getsize(file_path)
                t = tqdm(total=filesize)

                read_chunk = 1024 * 1024 * 32  # Adjust according to memory
                self.examples = []
                for text in read_in_chunks(file_path, chunk_size=read_chunk):
                    # self.examples = []
                    text_chunks = chunks(text, 300000)
                    for chunk in text_chunks:
                        batch = tokenizer(
                            chunk, truncation=True, padding='max_length', return_overflowing_tokens=True)
                        for ids in batch['input_ids']:
                            self.examples.append(ids)
                    t.update(read_chunk)

                with open(cached_features_file, "ab") as handle:
                    pickle.dump(self.examples, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        # pickling for salient span masking
        elif args.mask_way == 'ssm':
            cached_features_file = os.path.join(
                directory, model_type_string +
                           "_cached_SSM_" + filename)
            if os.path.exists(cached_features_file):
                print("Loading features from cached file %s", cached_features_file)
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                print("Creating features from dataset file at %s", directory)

                tokenizer.save_vocabulary(save_directory='.')

                print('---' * 10)
                print("Saving features into cached file %s", cached_features_file)
                model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
                tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
                ner = pipeline('ner', model=model, tokenizer=tokenizer_ner)
                filesize = os.path.getsize(file_path)
                t = tqdm(total=filesize)

                read_chunk = 1024 * 1024 * 1
                self.examples = []
                for text in read_in_chunks(file_path, chunk_size=read_chunk):
                    # self.examples = []
                    text_chunks = chunks(text, 30000)
                    for chunk in text_chunks:
                        batch = tokenizer(
                            chunk, truncation=True, padding='max_length', return_overflowing_tokens=True,
                            max_length=510, stride=32, return_offsets_mapping=True)  # , add_special_tokens=False
                        for sent in batch['input_ids']:
                            sent_tokens = tokenizer.decode(sent, skip_special_tokens=True)
                            ner_tokens = tokenizer_ner(sent_tokens, return_offsets_mapping=True,
                                                       add_special_tokens=False)
                            offset_mapping = ner_tokens.pop("offset_mapping")
                            ner_tokens = tokenizer_ner.convert_ids_to_tokens(ner_tokens.pop("input_ids"))
                            # use bert-base-NER to identify named entities
                            ner_doc = ner(sent_tokens)
                            idx_lst = [x['index'] - 1 for x in ner_doc]
                            idx_lst = set(idx_lst)
                            entities_lst = []
                            i = 0
                            # Determine the span of the named entity in the token sequence of T5.
                            while i < len(ner_tokens):
                                if '##' not in ner_tokens[i]:
                                    if i in idx_lst and (i - 1) not in idx_lst:
                                        start_ind, end_ind = offset_mapping[i]
                                        offset = 1
                                        entities_lst.append((i, offset, start_ind, end_ind))
                                    if i in idx_lst and (i - 1) in idx_lst:
                                        _, end_ind = offset_mapping[i]
                                        idx, offset, start_ind, _ = entities_lst[-1]
                                        entities_lst[-1] = (idx, offset + 1, start_ind, end_ind)
                                    i += 1
                                else:
                                    _, end_ind = offset_mapping[i]
                                    if (i - 1) in idx_lst:
                                        idx_lst.add(i)
                                        idx, offset, start_ind, _ = entities_lst[-1]
                                        entities_lst[-1] = (idx, offset + 1, start_ind, end_ind)
                                    elif (i - 1) not in idx_lst and i in idx_lst:
                                        idx_lst.remove(i)
                                    i += 1
                            sent_ids = tokenizer(sent_tokens, return_offsets_mapping=True)
                            tokens_range = sent_ids['offset_mapping']
                            input_ids = sent_ids['input_ids']
                            if input_ids[-1] != 1:
                                print('the last element of input_ids is not 1')
                                sys.exit()
                            if input_ids[0] == 3:
                                input_ids = input_ids[1:]
                                tokens_range = tokens_range[1:]
                            tokens_range.pop(-1)
                            i = 0
                            entities_idx = []
                            # Organize the IDs of the named entities
                            for entity_idx in entities_lst:
                                continuous = False
                                while i < len(input_ids) - 1:
                                    if tokens_range[i][0] == entity_idx[2] and tokens_range[i][1] == entity_idx[-1]:
                                        entities_idx.append([i])
                                        if i + 1 != (len(input_ids) - 1):
                                            if tokens_range[i] == tokens_range[i + 1]:
                                                entities_idx.append([i + 1])
                                                i += 2
                                                break
                                        i += 1
                                        break
                                    elif tokens_range[i][0] == entity_idx[2] and tokens_range[i][1] != entity_idx[-1]:
                                        entities_idx.append([i])
                                        continuous = True
                                    elif continuous and tokens_range[i][1] < entity_idx[-1]:
                                        entities_idx[-1].append(i)
                                    elif continuous and tokens_range[i][1] == entity_idx[-1]:
                                        entities_idx[-1].append(i)
                                        if i + 1 != (len(input_ids) - 1):
                                            if tokens_range[i] == tokens_range[i + 1]:
                                                entities_idx[-1].append(i + 1)
                                                i += 2
                                                break
                                        i += 1
                                        break
                                    i += 1
                            en_ids = [k for x in entities_idx for k in x]
                            others_idx = list(set(range(len(input_ids[:-1]))).difference(set(en_ids)))
                            # print(len(entities_idx), entities_idx)
                            # print(len(others_idx), others_idx)
                            entities_num = sum([len(x) for x in entities_idx])
                            if entities_num + len(others_idx) != (len(input_ids) - 1):
                                print(entities_num + len(others_idx), len(input_ids) - 1)
                                print('Error Error!')
                                sys.exit()
                            if len(input_ids) < 4:
                                print('This sample length < 2')
                                continue
                            input_ids += [0] * (512 - len(input_ids))
                            # Supplement the vacancy tokens caused by the difference between the tokenization of T5 and BERT.
                            if len(input_ids) != 512:
                                print('input_ids != 512')
                                sys.exit()
                            self.examples.append([input_ids, entities_idx, others_idx])
                    t.update(read_chunk)
                with open(cached_features_file, "ab") as handle:
                    pickle.dump(self.examples, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        # pickling for PMI-masking
        elif args.mask_way == 'pmi':
            cached_features_file = os.path.join(
                directory, model_type_string +
                           "_cached_PMI_" + str(block_size) + "_" + filename)

            if os.path.exists(cached_features_file):
                print("Loading features from cached file %s", cached_features_file)
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                print("Creating features from dataset file at %s", directory)
                tokenizer.save_vocabulary(save_directory='.')
                print('---' * 10)
                print("Saving features into cached file %s", cached_features_file)
                filesize = os.path.getsize(file_path)
                t = tqdm(total=filesize)
                assert os.path.exists(args.pmi_path)
                with open(args.pmi_path, "rb") as handle:
                    pmi_vocab = pickle.load(handle)[:800000]
                read_chunk = 1024 * 1024 * 32  # Adjust according to memory
                self.examples = []
                for text in read_in_chunks(file_path, chunk_size=read_chunk):
                    # self.examples = []
                    text_chunks = chunks(text, 300000)
                    for chunk in text_chunks:
                        batch = tokenizer(chunk, truncation=True, return_overflowing_tokens=True)#, padding='max_length'
                        for ids in batch['input_ids']:
                            #ids = list(filter(lambda x: x != tokenizer.pad_token_id, ids))
                            if ids[-1] != 1:
                                print('the last element of input_ids is not 1')
                                sys.exit()
                            if ids[0] == 3:
                                ids = ids[1:]
                            pmi_ids, others_ids = pmi_tokens(ids,  pmi_vocab, tokenizer.pad_token_id)
                            ids += [0] * (512 - len(ids))
                            if len(ids) != 512:
                                print('input_ids != 512')
                                sys.exit()
                            self.examples.append([ids, pmi_ids, others_ids])
                    t.update(read_chunk)

                with open(cached_features_file, "ab") as handle:
                    pickle.dump(self.examples, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.args.mask_way == 'ssm' or self.args.mask_way == 'pmi':
            return [torch.tensor(self.examples[item][0], dtype=torch.long), self.examples[item][1],
                    self.examples[item][2]]
        else:
            return torch.tensor(self.examples[item], dtype=torch.long)


# if __name__ == '__main__':
#     tokenizer = AutoTokenizer.from_pretrained('t5-base')
#     args = {'model_type': 't5-base', 'mask_way': 'pmi',
#             'pmi_path': 'A:/thesis/knowledge-probing-private-tianyi/knowledge-probing-private-tianyi/knowledge_probing/test_files/pmi_dict_2000k_M.pkl'}
#     args = DotMap(args)
#     text = TextDataset(tokenizer=tokenizer, args=args, file_path='A:/thesis/data/paq_train.tsv', )
