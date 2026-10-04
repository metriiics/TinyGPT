import torch
from torch.utils.data import Dataset
import numpy as np


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt).ids

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


class BinDataset(Dataset):
    def __init__(self, filename, max_length, stride):
        self.data = np.memmap(
            filename,
            dtype=np.uint16,
            mode="r"
        )

        self.context_length = max_length
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.context_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride

        x = self.data[start:start + self.context_length]
        y = self.data[start + 1:start + self.context_length + 1]

        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(y.astype(np.int64))
        )