import torch 
import torch.nn as nn 

class Embedding_Block(nn.Module): 
    def __init__(self, d_model: int, vocab_size: int, max_len: int):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.d_model)
        self.positional_embedding = nn.Embedding(num_embeddings = self.max_len, embedding_dim = self.d_model)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        pos_ids = torch.arange(start = 0, end = seq_len, device = x.device).unsqueeze(0)
        return self.token_embedding(x) + self.positional_embedding(pos_ids)



class Feed_Forward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        layers = []

        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(d_ff, d_model))

        self.ff = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.ff(x)