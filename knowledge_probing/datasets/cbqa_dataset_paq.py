from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import pickle
from tqdm import tqdm
import torch
import pandas as pd



class cbqa_Dataset(Dataset):
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
            print("Loading features from cached fine-tuning file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            print("Creating features from fine-tuning dataset file at %s", directory)

            tokenizer.save_vocabulary(save_directory='.')

            print('---'*10)
            print("Saving features into cached file %s", cached_features_file)

            with pd.read_json(file_path, encoding="utf_8", lines=True, chunksize=10000) as f:
                for data in f:
                    data = data.values.tolist()
                    self.examples = []
                    data_batch = []
                    for datum in data:
                        chunk = 'Closed-book exam: ' + datum[0] + ' <extra_id_0> ' + datum[1]
                        data_batch.append(chunk)
                    batch = tokenizer(data_batch, truncation=True,
                                      return_overflowing_tokens=True)#padding='max_length', 
                    for ids in batch['input_ids']:
                        self.examples.append(ids)
                    with open(cached_features_file, "ab") as handle:
                        pickle.dump(self.examples, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
