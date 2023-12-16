import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data_dir = Path('./datas')


class MyDatasets(Dataset):
    def __init__(self, data_path, device, pad_id=0):
        super(MyDatasets, self).__init__()
        self.PAD = pad_id
        self.data = pickle.load(open(data_path, 'rb'))
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x, y = self.data[item]
        return x, y, len(x)

    def collate_fn(self, batch):
        x, y, lengths = list(zip(*batch))
        max_length = max(lengths)
        mask = np.zeros(shape=(len(batch), max_length))
        for i, length in enumerate(lengths):
            x[i].extend([self.PAD] * (max_length - length))
            mask[i, :length] = 1
        # print(batch)
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        mask = torch.from_numpy(mask)
        return x, y, mask


def creater_dataloader(data_path, batch_size, device='cpu', shuffle=True):
    datasets = MyDatasets(data_path, device)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True, collate_fn=datasets.collate_fn)
    return dataloader


if __name__ == '__main__':
    dataloader = creater_dataloader(str(data_dir / 'train_data.pkl'), batch_size=32)
    for x, y, mask in dataloader:
        print(x, y, mask)
        break
