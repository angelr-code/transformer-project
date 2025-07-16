import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math

class EmbeddingBlock(nn.Module): 
    """
    Implements the tokens embeddings and the positional encoding.

    Parameters
    ----------
    d_model: int 
        Output dimensions of the model sub-layers and embedding.
    
    vocab_size: int 
        Size of the tokens dictionary.

    max_len: int 
        Maximum sequence length admitted.
    """
    def __init__(self, d_model: int, vocab_size: int, max_len: int):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.d_model)
        self.positional_encoding = nn.Embedding(num_embeddings = self.max_len, embedding_dim = self.d_model)   # Learnable Positional Encoding. In the original transformer paper: Sinusoidal and constant.

    def forward(self, x: torch.Tensor):
        """
        Computes the tokens embeddings.
        
        Parameters
        ----------
        x: torch.Tensor
            Input to be embedded

        Returns
        -------
        torch.Tensor
            The tokenized sentence embeddings
        """
        seq_len = x.size(1)
        pos_ids = torch.arange(start = 0, end = seq_len, device = x.device).unsqueeze(0)
        return self.token_embedding(x) + self.positional_encoding(pos_ids)




class FeedForward(nn.Module):
    """
    Feed Forward transformer block. Consists of a fully connected Neural Network with ReLU activation.

    Parameters
    ----------
    d_model: int
        Output dimensions of the model sub-layers and embedding.

    d_ff: int
        Hidden layer dimension.

    dropout: float
        Probability of applying dropout to an artificial neuron.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        layers = []
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.GELU()) # GELU activation function (Gaussian CDF). In the original trasnformer paper: ReLU.
        layers.append(nn.Linear(d_ff, d_model))

        self.ff = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Computes the forward pass thourgh the network.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor.
        
        Returns
        -------
        torch.Tensor
            The Neural Network output.
        """

        return self.ff(x)
    



class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Block. 

    Parameters
    ----------
    d_model: int
        Output dimensions of the model sub-layers and embedding.

    heads: int
        Number of attention heads.

    dropout: float
        Probability of applying dropout to an artificial neuron.
    """

    
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        if d_model % heads != 0:
            raise ValueError("The model dimension, d_model, must be divisible by the number of attention heads.")
        
        self.d_k = d_model // heads # Attention heads projection dimensions of queries and keys (and usually values). 
        
        # Weight Matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final projections after heads concatenation
        self.w_o = nn.Linear(d_model, d_model)


    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask):
        """
        Computes scaled dot-product attention.

        Parameters
        ----------
        Q : torch.Tensor
            Query tensor of shape (batch_size, num_heads, seq_len, d_k).

        K : torch.Tensor
            Key tensor of shape (batch_size, num_heads, seq_len, d_k).

        V : torch.Tensor
            Value tensor of shape (batch_size, num_heads, seq_len, d_k).

        mask : torch.Tensor or None
            Optional mask tensor to prevent attention to certain positions.
            Should be broadcastable to (batch_size, num_heads, seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Attention output of shape (batch_size, seq_len, d_model).
        """

        batch_size = Q.size(0)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim = -1) @ V

        #Finally we concat the attention heads results
        attn_output = attention.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        return attn_output
    

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        """
        Applies the multi-head attention mechanism to input tensors.

        Parameters
        ----------
        q : torch.Tensor
            Query input tensor of shape (batch_size, seq_len, d_model).

        k : torch.Tensor
            Key input tensor of shape (batch_size, seq_len, d_model).

        v : torch.Tensor
            Value input tensor of shape (batch_size, seq_len, d_model).

        mask : torch.Tensor or None
            Optional mask tensor to prevent attention to certain positions.
            Should be broadcastable to (batch_size, num_heads, seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Attention output of shape (batch_size, seq_len, d_model).
        """
        batch_size = q.size(0)

        # Linear projections
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # We group the tensors by heads to use each independent position
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn = self.attention(self, Q, K, V, mask)
 
        return self.w_o(attn)
    


class LayerNorm(nn.Module):
    """
    Layer Normalization with learnable parameters.

    Parameters
    ----------
    d_model: int
        Output dimensions of the model sub-layers and embedding.

    eps: float 
        Avoids dividing by 0 in the Normalization step.
    """ 

    def __init__(self, d_model: int, eps = 10**-6):
        super().__init__()
        self.eps = eps 

        # We define the scaling and location as PyTorch trainable parameters of the model.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        """
        Computes the Layer Normalization term. 

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor.

        Returns
        -------
        layernorm: torch.Tensor
            Layer-Normalized Tensor.
        """

        # We calculate the mean and std of each sample independently.
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        layernorm = self.gamma * (x - mean)/(std + self.eps) + self.beta

        return layernorm 




class ResidualConnection(nn.Module):
        """
        Transformers residual connection in the Add & Norm Block.

        Parameters
        ----------
        d_model: int
            Output dimensions of the model sub-layers and embedding.

        eps: float 
            Avoids dividing by 0 in the Normalization step.
        """

        def __init__(self, d_model: int, eps = 10**-6):
            super().__init__()
            self.layernorm = LayerNorm(d_model, eps)


        def forward(self, x, sublayer):
            """
            Computes the residual connection by adding residual data and its layer
            normalized expression after going through the previous sublayer block.

            Parameters
            ----------
            x: torch.Tensor
                Input Tensor. Before applying sublayer operations.

            Returns
            -------
            addnorm: torch.Tensor
                Residual connection output.
            """

            addnorm = x + sublayer(self.norm(x)) # Pre-norm improvement. In the original transformer paper: x + sublayer(x) -> LayerNorm

            return addnorm