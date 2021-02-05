from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import pickle
from tqdm import tqdm
import torch
from knowledge_probing.datasets.text_data_utils import chunks
from datasets import load_dataset


class FullWikiTextDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, args, mode, block_size=512):
        # cache_dir = os.path.join(args.output_base_dir,
        #                          'training_data', 'full_wikipedia')
        cache_dir = args.full_wiki_cache_dir
        print('Saving cached full wiki to', cache_dir)
        assert os.path.isdir(cache_dir)

        block_size = block_size - \
            (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        model_type_string = args.model_type.replace('/', '-')
        cached_features_file = os.path.join(
            cache_dir, model_type_string +
            "_cached_lm_" + str(block_size) + '_fullwiki_' + mode
        )

        if os.path.exists(cached_features_file):
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            self.examples = []

            if mode == 'train':
                data = load_dataset(
                    "wikipedia", '20200501.en', split='train[:1%]')
                data = data['text'][: int(len(data['text']) * .1)]
            elif mode == 'valid':
                data = load_dataset(
                    "wikipedia", '20200501.en', split='train[1%:2%]')
                # Only use 10% of this array
                data = data['text'][: int(len(data['text']) * .01)]
            else:
                data = load_dataset(
                    "wikipedia", '20200501.en', split='train[2%:3%]')
                data = data['text'][: int(len(data['text']) * .01)]

            for wikipage in data:

                # Remove references and stuff from wikipedia page
                wikipage = self.clean_wikipage(wikipage)

                batch = tokenizer(
                    wikipage, truncation=True, padding='max_length', return_overflowing_tokens=True)

                for ids in batch['input_ids']:
                    self.examples.append(ids)

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def clean_wikipage(self, page: str) -> str:
        if '\nSee also\n' in page:
            rm_index = page.index('\nSee also\n')
            page = page[:rm_index]
        if '\nReferences\n' in page:
            rm_index = page.index('\nReferences\n')
            page = page[:rm_index]

        return page

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
