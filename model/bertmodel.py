import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class Preprocessor(nn.Module):
    def __init__(self, model_dir):
        super(Preprocessor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

    def forward(self, text):
        return self.tokenizer(text, padding='longest')


class BertTokenClassificy(nn.Module):
    def __init__(self, model_dir, num_labels):
        super(BertTokenClassificy, self).__init__()

        self.bert: BertModel = BertModel.from_pretrained(model_dir)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden_state = self.bert(input_ids, attention_mask, token_type_ids)
        logic_prob = self.linear(torch.mean(hidden_state.last_hidden_state, dim=1))
        logic_prob = self.dropout(logic_prob)
        return logic_prob
    



if __name__ == '__main__':
    model_dir = r'G:\models\bert-base-chinese'
    tokenizer = Preprocessor(model_dir=model_dir)
    text = ['还有双鸭山到淮阴的汽车票吗13号的',
            '从这里怎么回家']
    print(tokenizer(text))
