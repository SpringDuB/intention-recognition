import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import json

def _load_cdv(path, header=None, sep=r','):
    datas = pd.read_csv(path, sep=sep)
    return datas


def convert_text_to_ids(texts, tokenizer):
    return tokenizer(texts)


def create_tokenizer(model_dir):
    from model.bertmodel import Preprocessor
    return Preprocessor(model_dir)


def load_csv_to_list(path):
    labels = json.load(open('datas/labels_.json'))
    datas: DataFrame = _load_cdv(path)
    inputs = []
    for data in datas.values:
        if len(data) == 2:
            text = data[0]
            label = data[1]
            inputs.append([text, labels.get(label, None)])
        else:
            inputs.append([data[0], None])
    return inputs


class MyBertDataset(Dataset):
    def __init__(self, model_dir, path):
        super(MyBertDataset, self).__init__()
        self.tokenizer = create_tokenizer(model_dir)
        self.datas = load_csv_to_list(path)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        text, label = self.datas[item]
        return text, label

    def collate_fn(self, batch):
        texts = []
        labels = []
        for text, label in batch:
            texts.append(text)
            labels.append(label)
        return convert_text_to_ids(texts, self.tokenizer), labels


def creat_dataloader(model_dir, path, batch_size, shuffle=False, num_workers=0):
    dataset = MyBertDataset(model_dir, path)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=dataset.collate_fn
    )


if __name__ == '__main__':
    model_dir = r'/mnt/workspace/nlp/models/bert-base-chinese'
    path = 'datas/split_zh_train_data.csv'
    dataloader = creat_dataloader(model_dir, path, batch_size=12)
    for input, label in dataloader:
        print(input, label)
        break
    # data = trans_texts_to_ids(model_dir, path)

