import json
import pickle
import re
from pathlib import Path
import warnings
import pandas as pd
import jieba
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings(action='ignore')

data_path = Path('../datas')
data_path.mkdir(parents=True, exist_ok=True)


def is_num(token):
    return re.match(r'^[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$', token)


def is_pun(token):
    if token in [',', '，', ':', ' ', '.', '。']:
        return True
    return False


def stage1():
    data = pd.read_csv(str(data_path / 'train.csv'), sep=r'\t+', header=None)
    # print(data)
    texts, labels = [], []
    token_count = dict()
    labelss = set()
    for text, label in tqdm(data.values):
        # print(text, label)
        text = text.strip()
        text = jieba.lcut(text)
        for token in text:
            token_count[token] = token_count.get(token, 0) + 1
        texts.append(' '.join(text))
        labels.append(label)
        labelss.add(label)
    df = pd.DataFrame((texts, labels)).T
    df.to_csv(str(data_path / 'split_data.csv'), index=False)
    token_count = list(token_count.items())
    token_count.sort(key=lambda t: -t[1])
    token_count = [token for token, count in token_count if count >= 2]
    for token in ['UKN', 'NUM', 'PUN']:
        token_count.insert(0, token)
    json.dump(token_count, open(str(data_path / 'token_count.json'), 'w', encoding='utf-8'), indent=3,
              ensure_ascii=False)
    labelss = list(labelss)
    save_lable(labelss, str(data_path / 'labels_ids.json'))
    reverse_labels(labelss, str(data_path / 'ids2lables.json'))
    json.dump(labelss, open(str(data_path / 'labels.json'), 'w', encoding='utf-8'), ensure_ascii=False)


def stage2():
    tokens = json.load(open(str(data_path / 'token_count.json'), 'r', encoding='utf-8'))
    labels = json.load(open(str(data_path / 'labels.json'), 'r', encoding='utf-8'))
    split_data = pd.read_csv(str(data_path / 'split_data.csv'), sep=',')
    split_data.columns = ['text', 'label']
    print(split_data)
    print(tokens[:10])
    token2ids = dict()
    label2id = dict()
    for token in tokens:
        token2ids[token] = len(token2ids)
    for label in labels:
        label2id[label] = len(label2id)
    # print(token2ids)
    # print(label2id)
    sen_vec = []
    for text, label in tqdm(split_data.values):
        tokensids = []

        for token in text.split(' '):
            if is_num(token):
                tokensids.append(token2ids['NUM'])
            elif is_pun(token):
                tokensids.append(token2ids['PUN'])
            else:
                tokensids.append(token2ids.get(token, 0))
        sen_vec.append([tokensids, label2id[label]])
    print(sen_vec[:10])
    train_data, test_data = train_test_split(sen_vec, test_size=0.25, random_state=22, shuffle=True)
    print(len(train_data), len(test_data))
    # pickle.dump(train_data, open(str(data_path / 'train_data.pkl'), 'wb'))
    # pickle.dump(test_data, open(str(data_path / 'test_data.pkl'), 'wb'))

    pd.DataFrame(train_data).to_csv(str(data_path / 'train_data.csv'))
    pd.DataFrame(test_data).to_csv(str(data_path / 'test_data.csv'))


def load_data():
    data = pickle.load(open(str(data_path / 'train_data.pkl'), 'rb'))
    # print(data)
    for ids, label in data:
        print(ids, label)
        break


def load_csv():
    datas = pd.read_csv(r'datas/train_data.csv', header=None)
    for data in datas:
        print(data)
        break


def save_lable(datas, path):
    labels = {}
    for data in datas:
        labels[data] = len(labels)
    json.dump(labels, open(path, 'w', encoding='utf-8'))
    print('保存labels成功')


def reverse_labels(datas, path):
    labels = {}
    for i, data in enumerate(datas):
        labels[int(i)] = data
    json.dump(labels, open(path, 'w', encoding='utf-8'))
    print('保存ids2labels成功')


if __name__ == '__main__':
    stage1()
    stage2()
    load_csv()
