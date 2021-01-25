from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import pickle
from tqdm import tqdm
import torch
from knowledge_probing.datasets.text_data_utils import chunks


class TextDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, args, file_path: str, block_size=512):
        print(file_path)
        assert os.path.isfile(file_path)

        block_size = block_size - \
            (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        model_type_string = args.model_type.replace('/', '-')
        cached_features_file = os.path.join(
            directory, model_type_string +
            "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file):
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            print("Creating features from dataset file at %s", directory)

            tokenizer.save_vocabulary(save_directory='.')

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            text_chunks = list(chunks(text, 300000))

            for chunk in tqdm(text_chunks):
                batch = tokenizer(
                    chunk, truncation=True, padding='max_length', return_overflowing_tokens=True)

                for ids in batch['input_ids']:
                    self.examples.append(ids)

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
