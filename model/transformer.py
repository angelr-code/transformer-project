import torch 
import torch.nn as nn 
import torch.nn.functional as F
from layers import EmbeddingBlock, EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    """
    Defines a transformer encoder block.

    Parameters 
    ----------
    layers: int
        Number of encoder layers in the model.

    d_model: int
        Output dimensions of the model sub-layers and embedding.

    vocab_size: int
        Size of the tokens dictionary.

    max_len: int
        Maximum length for sequences

    d_ff: int
        Hidden layer dimension.
    
    heads: int
        Number of attention heads.

    dropout: float
        Probability of applying dropout to an artificial neuron.
        Default is 0.1.

    eps: float 
        Avoids dividing by 0 in the Normalization step.
        Default is 10^-6.
    """

    def __init__(self, layers: int, d_model: int, vocab_size: int, max_len: int, d_ff:int, heads: int, dropout = 0.1, eps = 10**-6):  
        super().__init__()

        self.embedding = EmbeddingBlock(d_model, vocab_size, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, heads, dropout, eps)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, mask = None):
        """
        Computes the output of the Encoder.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        mask : torch.Tensor or None
            Optional mask tensor to prevent attention to certain positions.
            Should be broadcastable to (batch_size, num_heads, seq_len, seq_len).
            In the Encoder layer default is None.

        Returns
        -------
        torch.Tensor
            Encoder output tensor.
        """

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class Decoder(nn.Module):
    """
    Defines a transformer decoder block.

    Parameters 
    ----------
    layers: int
        Number of encoder layers in the model.

    d_model: int
        Output dimensions of the model sub-layers and embedding.

    vocab_size: int
        Size of the tokens dictionary.

    max_len: int
        Maximum length for sequences

    d_ff: int
        Hidden layer dimension.
    
    heads: int
        Number of attention heads.

    dropout: float
        Probability of applying dropout to an artificial neuron.
        Default is 0.1.

    eps: float 
        Avoids dividing by 0 in the Normalization step.
        Default is 10^-6.
    """

    def __init__(self, layers: int, d_model: int, vocab_size: int, max_len: int, d_ff:int, heads: int, dropout = 0.1, eps = 10**-6):
        super().__init__()

        self.embedding = EmbeddingBlock(d_model, vocab_size, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, heads, dropout, eps) 
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor, tgt_mask: torch.Tensor, memory_mask = None):
        """
        Computes the output of the Decoder.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        x_encoder: torch.Tensor
            Encoder output.

        tgt_mask : torch.Tensor or None
            Mask tensor to prevent attention to certain positions in the masked
            multi-head attention block.
            Should be broadcastable to (batch_size, num_heads, seq_len, seq_len).

        memory_mask: torch.Tensor or None
            Optional mask tensor to prevent attention to certain positions in the
            encoder-decoder attention block.Should be broadcastable to 
            (batch_size, num_heads, seq_len, seq_len). Default is None.

        Returns
        -------
        torch.Tensor
            Encoder output tensor.
        """
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, x_encoder, tgt_mask, memory_mask)
            
        return x