import json

import torch

from utils.tokenizer_process import create_tokenizer


class Predict(object):
    def __init__(self, model_path, tokenizer_dir, labels_path):
        super(Predict, self).__init__()
        self.tokenizer = create_tokenizer(tokenizer_dir)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.labels = json.load(open(labels_path))

    def predict(self, text):
        id2labels = {}
        for label, value in self.labels.items():
            id2labels[value] = label
        input = self.tokenizer(text)
        input_ids = torch.unsqueeze(torch.tensor(input.input_ids, dtype=torch.long), dim=0).to(self.device)
        attention_mask = torch.unsqueeze(torch.tensor(input.attention_mask, dtype=torch.float), dim=0).to(self.device)
        token_type_ids = torch.unsqueeze(torch.tensor(input.token_type_ids, dtype=torch.long), dim=0).to(self.device)
        output = self.model(input_ids, attention_mask, token_type_ids)
        pred = torch.argmax(output, dim=-1)
        res = {
            'text': text,
            'predict': id2labels[int(pred)]
        }
        return res


if __name__ == '__main__':
    text = "我想看挑战两把s686打突变团竞的游戏视频"
    model_path = r'model/datas/model/best_model.pt'
    tokenizer_dir = r'G:\models\bert-base-chinese'
    label_path = r'datas/labels_map.json'
    res = Predict(model_path, tokenizer_dir, label_path).predict(text)
    print(res)
