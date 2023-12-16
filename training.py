import os

from utils.bert_trainer import BertTrainer
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch import optim
from tqdm import tqdm

from utils.args_config import read_model_args
from model.LSTM_model import LstmModel
from utils.utils import creater_dataloader

data_dir = Path('./datas')


# def run(train_data_path, eval_data_path,
#         batch_size, vocab_size,
#         embedding_dim, class_num, epochs, output_dir, device):
#     # 创建输出目录
#     output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
#     log_dir = Path(str(output_dir / 'log'))
#     model_dir = Path(str(output_dir / 'model'))
#     log_dir.mkdir(parents=True, exist_ok=True)
#     model_dir.mkdir(parents=True, exist_ok=True)
#     # 数据加载
#     train_datalader = creater_dataloader(train_data_path, batch_size, device)
#     eval_dataloader = creater_dataloader(eval_data_path, batch_size, device)
#     net = LstmModel(vocab_size=vocab_size, embedding_dim=embedding_dim, class_num=class_num)
#     model_list = os.listdir(str(model_dir))
#     if len(model_list):
#         model_list.sort(key=lambda t: t.split('_')[1].split('.')[0])
#         model_name = model_list[-1]
#         ori_net = torch.load(str(model_dir / model_name))
#         net.load_state_dict(ori_net.state_dict())
#
#     opt = optim.Adam(net.parameters(), lr=0.001)
#     loss_fn = nn.CrossEntropyLoss()
#     if torch.cuda.is_available() and device == 'cuda:0':
#         net.to(device)
#     else:
#         net.to(device)
#     for epoch in range(epochs):
#         # 模型训练
#         train_bar = tqdm(train_datalader)
#         for x, y, mask in train_bar:
#             opt.zero_grad()
#             output = net(x, mask=None)
#             loss = loss_fn(output, y)
#             loss.backward()
#             opt.step()
#             train_bar.set_postfix(epoch=epoch, loss=loss.item())
#         # 模型测试
#         pred = []
#         true = []
#         net.eval()
#         with torch.no_grad():
#             for x, y, mask in tqdm(eval_dataloader):
#                 output = net(x, mask=None)
#                 predict = torch.argmax(output, dim=1)
#                 pred.append(predict)
#                 true.append(y)
#             pred = torch.concat(pred, dim=0).to('cpu').numpy()
#             true = torch.concat(true, dim=0).to('cpu').numpy()
#             print(f'正确率{accuracy_score(pred, true)}')
#         # 模型保存
#         if epoch % 10 == 0:
#             torch.save(net, str(model_dir / f'net_{epoch}.pkl'))
#         train_bar.close()
#     torch.save(net, str(model_dir / f'net_{epochs}.pkl'))


def begin_train(args):
    trainer = BertTrainer(args)
    trainer.train()

    
if __name__ == '__main__':
    # labels = json.load(open(str(data_dir / 'labels_.json'), 'r', encoding='utf-8'))
    # class_num = len(labels)
    # tokens = json.load(open(str(data_dir / 'token_count.json'), 'r', encoding='utf-8'))
    # vocab_size = len(tokens)
    # train_data_path = str(data_dir / 'train_data.pkl')
    # eval_data_path = str(data_dir / 'test_data.pkl')
    # embedding_dim = 128
    # epochs = 30
    # batch_size = 32
    # output_dir = 'model/datas'
    # device = 'cuda:0'
    # run(train_data_path=train_data_path, eval_data_path=eval_data_path, batch_size=batch_size,
    #     vocab_size=vocab_size, embedding_dim=embedding_dim, class_num=class_num,
    #     epochs=epochs, output_dir=output_dir, device=device
    #     )
    args = read_model_args()
    args = args.parse_args()
    begin_train(args)
