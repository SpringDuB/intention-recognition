import argparse
import json

import pandas
from sklearn.model_selection import train_test_split


def split_data(args):
    path = args.path
    sep = args.sep
    header = args.header
    split_size = args.split_size
    output_dir = args.output_dir
    data = pandas.read_csv(path, header=header, sep=sep).values
    labels = set(data[:, 1])
    labels_dict = {}
    for label in labels:
        labels_dict[label] = len(labels_dict)
    id2labels = {}
    for label, value in labels_dict.items():
        id2labels[value] = label

    json.dump(labels_dict, open('datas/labels_map.json', 'w', encoding='utf-8'))
    json.dump(id2labels, open('datas/id2labels_map.json', 'w', encoding='utf-8'))
    print(labels_dict)
    train_data, val_data = train_test_split(data, train_size=split_size)
    pandas.DataFrame(train_data).to_csv(output_dir + '/split_train_data.csv', header=False, index=False)
    pandas.DataFrame(val_data).to_csv(output_dir + '/split_val_data.csv', header=False, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--path', required=True)
    args.add_argument('--split_size', default=0.75, type=int)
    args.add_argument('--sep', default=r'\t+')
    args.add_argument('--header', default=None)
    args.add_argument('--output_dir', default='datas')
    args = args.parse_args()
    split_data(args)
    print('successful')
