import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tokenizer_process import creat_dataloader
from model.bertmodel import BertTokenClassificy
from torch import optim
from torch import nn
import logging
from optimizer import build_optimizer
from torch.onnx import TrainingMode


class BertTrainer(object):
    def __init__(self, args):
        self.model_dir = args.model_dir
        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.batch_size = args.batch_size
        self.num_labels = args.labels_num
        self.epochs = args.epochs
        self.output_dir = args.output_dir
        self.labels_map = args.labels_map
        self.shuffle = args.shuffle
        self.num_workers = args.num_workers
        self.save_size = args.save_size
        self.train_dataloader = creat_dataloader(self.model_dir, self.train_data_path, self.batch_size, self.shuffle, self.num_workers)
        self.test_dataloader = creat_dataloader(self.model_dir, self.test_data_path, self.batch_size, self.shuffle, self.num_workers)
        self.model = BertTokenClassificy(self.model_dir, self.num_labels)
        # self.opt = optim.Adam(self.model.parameters(), lr=0.0001)
        # for n,p in self.model.named_parameters():
        #     print(n)
        self.opt = build_optimizer(self.model)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.model.to(device)
        pre = 0
        num = 0
        best_model = None
        for epoch in range(self.epochs):
            # 模型训练
            pred = []
            true = []
            train_bar = tqdm(self.train_dataloader, ncols=80)
            for x, y in train_bar:
                input_ids = torch.tensor(x['input_ids'], dtype=torch.long).to(device)
                attention_mask = torch.tensor(x['attention_mask'], dtype=torch.float).to(device)
                token_type_ids = torch.tensor(x['token_type_ids'], dtype=torch.long).to(device)
                y = torch.tensor(y, dtype=torch.long).to(device)
                self.opt.zero_grad()
                output = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.loss_fn(output, y)
                loss.backward()
                self.opt.step()
                train_bar.set_postfix(epoch=epoch, loss=loss.item())
                predict = torch.argmax(output, dim=1)
                pred.append(predict)
                true.append(y)
            pred = torch.concat(pred, dim=0).to('cpu').numpy()
            true = torch.concat(true, dim=0).to('cpu').numpy()
            print(f'正确率{accuracy_score(pred, true)}')
            # 模型测试
            pred = []
            true = []
            self.model.eval()
            with torch.no_grad():
                for x, y in tqdm(self.test_dataloader, ncols=80):
                    input_ids = torch.tensor(x['input_ids'], dtype=torch.long).to(device)
                    attention_mask = torch.tensor(x['attention_mask'], dtype=torch.float).to(device)
                    token_type_ids = torch.tensor(x['token_type_ids'], dtype=torch.long).to(device)
                    y = torch.tensor(y, dtype=torch.long).to(device)
                    output = self.model(input_ids, attention_mask, token_type_ids)
                    predict = torch.argmax(output, dim=1)
                    pred.append(predict)
                    true.append(y)
                pred = torch.concat(pred, dim=0).to('cpu').numpy()
                true = torch.concat(true, dim=0).to('cpu').numpy()
                score = accuracy_score(pred, true)
                print(f'正确率{score}')
                if score >= pre:
                    pre = score
                    best_model = self.model.state_dict
                else:
                    num += 1
            # 模型保存    
            if epoch % self.save_size == 0:
                path = self.output_dir + '/' + f'traced_bert_{epoch}.pt'
                dummy_tokens_tensor = torch.randint(10, size=(1, 20)).to(device)
                dummy_attention_mask = torch.randn(size=(1, 20)).to(device)
                dummy_token_type_ids = torch.randint(1, size=(1, 20)).to(device)
                traced_model = torch.jit.trace(
                    self.model,
                    [dummy_tokens_tensor, dummy_attention_mask,dummy_token_type_ids]
                )
                torch.jit.save(traced_model, path)
            if num > 4:
                break
            train_bar.close()
        path = self.output_dir + '/' + 'best_model.pt'
        dummy_tokens_tensor = torch.randint(10, size=(1, 20)).to(device)
        dummy_attention_mask = torch.randn(size=(1, 20)).to(device)
        dummy_token_type_ids = torch.randint(1, size=(1, 20)).to(device)
        traced_model = torch.jit.trace(
            self.model,
            [dummy_tokens_tensor, dummy_attention_mask, dummy_token_type_ids]
        )
        torch.jit.save(traced_model, path)



    
            
            
            