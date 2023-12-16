import torch
import torch.nn as nn


class LstmModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_num=12):
        super(LstmModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.feature = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(in_features=embedding_dim, out_features=class_num)

    def forward(self, x, mask=None):
        z1 = self.emb(x)  # [N, T, E]
        z2 = self.feature(z1)  # [N, T, 2E]
        if mask is not None:
            length = torch.sum(mask, dim=1, dtype=torch.long, keepdim=True)
            mask = torch.unsqueeze(mask, dim=2)
            z2 = z2 * mask
            z2 = (torch.sum(z2, dim=1) / length).to(torch.float32)
        else:
            z2 = torch.mean(z2, dim=1)
        z3 = self.fc1(z2)
        return z3
