from typing import Tuple
from torch.utils.data import Dataset
import torch
from knowledge_probing.datasets.text_data_utils import datasets_handle
from transformers import AutoTokenizer


# Noted that before using MultitaskDataset, please use TextDataset to pickling each data set separately.
class MultitaskDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, args, file_path: str, block_size=512):
        #self.args = args
        file_paths = file_path.split('&&')
        samples_set = []
        for file in file_paths:
            examples = datasets_handle(file, args, tokenizer)
            samples_set.append(examples)
        # Sort according to the size of the dataset
        if len(samples_set[0]) >= len(samples_set[1]):
            set_one = samples_set[0]
            set_sign1 = 'text'
            set_two = samples_set[1]
            set_sign2 = 'QA'
        else:
            set_one = samples_set[1]
            set_two = samples_set[0]
            set_sign1 = 'QA'
            set_sign2 = 'text'
        self.examples = []
        back_up = set_two.copy()
        batch_size = 32
        # Alternately arrange the samples of the 2 tasks according to a certain batch size.
        # The lesser one will be recycled.
        while set_one:
            for _ in range(batch_size):
                if not set_two:
                    set_two = back_up.copy()
                self.examples.append((set_sign2, set_two.pop()))
            for _ in range(batch_size):
                if not set_one:
                    break
                self.examples.append((set_sign1, set_one.pop()))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if len(self.examples[item][1]) == 3:
            return [self.examples[item][0], [torch.tensor(self.examples[item][1][0], dtype=torch.long),
                    self.examples[item][1][1], self.examples[item][1][2]]]
        else:
            return [self.examples[item][0], torch.tensor(self.examples[item][1], dtype=torch.long)]
