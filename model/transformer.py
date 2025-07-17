import torch 
import torch.nn as nn 
import torch.nn.functional as F
from layers import EmbeddingBlock, FeedForward, MultiHeadAttention, ResidualConnection, EncoderLayer

class Encoder(nn.Module):
    def __init__(self, layers: int, d_model: int, vocab_size: int, max_len: int, d_ff:int, heads: int, dropout = 0.1, eps = 10**-6):
        super().__init__()

        self.embedding = EmbeddingBlock(d_model, vocab_size, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, heads, dropout, eps)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, mask = None):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
            
        return x

    