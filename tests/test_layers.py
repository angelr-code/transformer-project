import pytest
import torch 
from model.layers import EmbeddingBlock, FeedForward, MultiHeadAttention, ResidualConnection, LayerNorm

# Embedding Block tests

def test_embedding_output_shape():
    """
    Tests that the EmbeddingBlock returns the correct output shape 
    given an input tensor of token indices.
    """

    d_model = 512
    max_len = 50 
    vocab_size = 100
    emb = EmbeddingBlock(d_model, vocab_size, max_len)

    seq_len = 20
    batch_size = 2
    x = torch.randint(low = 0, high = vocab_size, size = (batch_size, seq_len))

    out = emb(x)

    assert out.shape == (batch_size, seq_len, d_model)


def test_positional_encoding():
    """
    Tests that positional encoding varies with position for tensors
    with the same tokens arranged differently.
    """
    d_model = 512
    max_len = 50 
    vocab_size = 100
    emb = EmbeddingBlock(d_model, max_len, vocab_size)

    x = torch.tensor([[1, 2, 3]])
    y = torch.tensor([[3, 1, 2]])

    assert not torch.allclose(emb(y), emb(x))


def test_embeddings_grad_flow():
    """
    Ensures that the embedding and positional encoding learnable
    parameters gradients flow normally in backpropagation.
    """
    d_model = 512
    max_len = 50 
    vocab_size = 100
    emb = EmbeddingBlock(d_model, vocab_size, max_len)

    seq_len = 20
    batch_size = 2
    x = torch.randint(low = 0, high = vocab_size, size = (batch_size, seq_len))

    out = emb(x)

    # An example of loss function
    loss = out.sum()
    loss.backward()

    assert emb.token_embedding.weight.grad is not None
    assert emb.positional_encoding.weight.grad is not None


def test_embedding_long_seq_error():
    """
    Checks if the model admits an input with a sequence length
    greater than max_length. It shouldn't.
    """
    d_model = 64
    max_len = 5 
    vocab_size = 100
    emb = EmbeddingBlock(d_model, max_len, vocab_size)

    x = torch.randint(low = 0, high = vocab_size, size = (1, 10))  # seq_len > max_len

    # We test if creating an embedding for a sequence length longer than max raises a exception (as it should)
    with pytest.raises(IndexError):
        emb(x)


def test_deterministic_embedding():
    """
    Tests if the embeddings are completely deterministic.
    """
    d_model = 512
    max_len = 50 
    vocab_size = 100
    emb = EmbeddingBlock(d_model, max_len, vocab_size)

    x = torch.tensor([[1, 2, 3]])
    
    out1 = emb(x)
    out2 = emb(x)

    assert torch.allclose(out1, out2)

# FeedForward Tests