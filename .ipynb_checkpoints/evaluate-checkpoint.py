import argparse

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from model.bertmodel import BertTokenClassificy
from tokenizer_process import creat_dataloader


def evaluate():
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', required=True)
    args.add_argument('--tokenizer_dir', required=True)
    args.add_argument('--eval_data_path', required=True)
    args.add_argument('--num_workers', default=0, type=int, help=""" 处理数据的进程数 """)
    args.add_argument('--num_labels', default=12,type=int)
    args.add_argument('--shuffle', default=True, help=""" 分割数据时是否进行打乱 """)
    args.add_argument('--batch_size', default=32, type=int, help=""" 训练batch的大小 """)
    args = args.parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = BertTokenClassificy(args.tokenizer_dir, args.num_labels)
    net = torch.load(args.model_path)
    model.load_state_dict(net.state_dict)
    eval_dataloader = creat_dataloader(
        args.tokenizer_dir,
        args.eval_data_path,
        args.batch_size,
        args.shuffle,
        args.num_workers
    )
    pred = []
    true = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(eval_dataloader, ncols=80):
            input_ids = torch.tensor(x['input_ids'], dtype=torch.long).to(device)
            attention_mask = torch.tensor(x['attention_mask'], dtype=torch.float).to(device)
            token_type_ids = torch.tensor(x['token_type_ids'], dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            predict = torch.argmax(output, dim=1)
            pred.append(predict)
            true.append(y)
        pred = torch.concat(pred, dim=0).to('cpu').numpy()
        true = torch.concat(true, dim=0).to('cpu').numpy()
        score = accuracy_score(pred, true)
        print(f'正确率{score}')


if __name__ == '__main__':
    evaluate()
