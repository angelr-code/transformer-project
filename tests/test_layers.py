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
    Tests if the embeddings are completely deterministic in inference mode,
    where dropout is disabled.
    """
    d_model = 512
    max_len = 50 
    vocab_size = 100
    emb = EmbeddingBlock(d_model, max_len, vocab_size)
    emb.eval()

    x = torch.tensor([[1, 2, 3]])
    
    out1 = emb(x)
    out2 = emb(x)

    assert torch.allclose(out1, out2)

# FeedForward Tests

def test_feedforward_output_shape():
    """
    Tests that the FeedForward block returns the correct output shape 
    given an input tensor.
    """

    d_model = 64
    d_ff = 128
    dropout = 0.2
    net = FeedForward(d_model, d_ff, dropout)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)

    out = net(x)

    assert out.shape == (batch_size, seq_len, d_model)

def test_feedforward_grad_flow():
    """
    Ensures that the Neural Network parameters gradients flow 
    correctly in backpropagation.
    """
    d_model = 64
    d_ff = 128
    dropout = 0.2
    net = FeedForward(d_model, d_ff, dropout)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)

    out = net(x)

    # An example of loss function
    loss = out.sum()
    loss.backward()

    for name, para in net.named_parameters():
        assert para.grad is not None, f"No gradients for: {name}"
        assert not torch.isnan(para.grad).any(), f"NaNs in gradients for: {name}"
        assert not torch.isinf(para.grad).any(), f"Inf in gradients for: {name}"

# Residual Connection and LayerNorm Tests

def test_residual_connection_output_shape():
    """
    Tests that the Residual Connection block returns the correct output shape 
    given an input tensor and a FeedForward Block.
    """
    d_model = 64
    d_ff = 128
    dropout = 0.2
    sublayer = FeedForward(d_model, d_ff, dropout)
    rescon = ResidualConnection(d_model)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)

    out = rescon(x, sublayer)

    assert out.shape == (batch_size, seq_len, d_model), "The output dimensions of the residual connections are wrong"


def test_residual_connection_grad_flow():
    """
    Tests that the Layer Normalization learnable parameters
    gradients flow correctly after applying a residual connection.
    """
    d_model = 64
    d_ff = 128
    dropout = 0.2
    sublayer = FeedForward(d_model, d_ff, dropout)
    rescon = ResidualConnection(d_model)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)

    out = rescon(x, sublayer)

    # An example loss function
    loss = out.sum()
    loss.backward()

    for name, para in rescon.layernorm.named_parameters():
        assert para.grad is not None, f"No gradients for: {name}"
        assert not torch.isnan(para.grad).any(), f"NaNs in gradients for: {name}"
        assert not torch.isinf(para.grad).any(), f"Inf in gradients for: {name}"

# Multi head Attention tests

def test_multi_head_attention_output_shape():
    """
    Tests that the Multi Head Attention block returns the correct output shape 
    given an input tensor.
    """
    d_model = 128
    heads = 4

    attention_block = MultiHeadAttention(d_model, heads)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)

    out = attention_block(x, x, x, mask = None)

    assert out.shape == (batch_size, seq_len, d_model), "The outputs dimensions of the Attention block are incorrect."


def test_multi_head_attention_head_dimensions():
    """
    Tests that when given there is an error if the number of attention
    heads does not divide the dimension d_model.
    """

    d_model = 128
    heads = 5

    with pytest.raises(ValueError):
        MultiHeadAttention(d_model, heads)


def test_multi_head_attention_grad_flow():
    """
    Tests that the Attention Weights gradients flow correctly through
    the parameters.
    """
    d_model = 128
    heads = 4
    attention_block = MultiHeadAttention(d_model, heads)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)

    out = attention_block(x, x, x, mask = None)

    # An example loss function
    loss = out.sum()
    loss.backward()

    weights = [attention_block.w_q, attention_block.w_k, attention_block.w_v, attention_block.w_v]

    for w in weights:
        for name, para in w.named_parameters():
            assert para.grad is not None, f"No gradients for: {name}"
            assert not torch.isnan(para.grad).any(), f"NaNs in gradients for: {name}"
            assert not torch.isinf(para.grad).any(), f"Inf in gradients for: {name}"

def test_attention_mask():
    """
    Checks that masking does not break the code.
    """
    d_model = 128
    heads = 4
    attention_block = MultiHeadAttention(d_model, heads)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, heads, seq_len, seq_len).bool()
    
    try:
        attention_block(x,x,x,mask)
    except Exception as e:
        assert False, f"Multi-head Attention failed with mask: {e}"

def test_multi_head_attention_determinism():
    """
    Ensures that in eval mode, when dropout is disabled, the attention 
    block returns deterministic outputs.
    """

    d_model = 128
    heads = 4
    attention_block = MultiHeadAttention(d_model, heads)

    batch_size = 2
    seq_len = 10 
    x = torch.rand(batch_size, seq_len, d_model)

    attention_block.eval()
    out1 = attention_block(x, x, x, mask = None)
    out2 = attention_block(x, x, x, mask = None)

    assert torch.allclose(out1, out2), "The attention block is not being deterministic."