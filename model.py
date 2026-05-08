import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF(nn.Module):
    """Radial basis function kernel module used by MMD-based losses.

    The module computes a multi-scale Gaussian RBF kernel matrix for a batch of
    input embeddings. If ``bandwidth`` is not provided, the bandwidth is
    estimated from the pairwise squared Euclidean distances in the current
    batch. Multiple bandwidth multipliers are summed to make the kernel less
    sensitive to a single scale choice.

    Parameters
    ----------
    n_kernels : int, default=5
        Number of Gaussian kernels with different bandwidth multipliers.
    mul_factor : float, default=2.0
        Multiplicative factor used to space the bandwidth scales.
    bandwidth : float or None, default=None
        Fixed kernel bandwidth. When ``None``, the bandwidth is estimated from
        the input batch during the forward pass.

    Notes
    -----
    ``forward(X)`` expects a tensor of shape ``(n_samples, n_features)`` and
    returns a kernel matrix of shape ``(n_samples, n_samples)``.
    """

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMD_loss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class OneLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(OneLayerMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.mlp(x)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def gelu(x):

    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class transformerModel(nn.Module):
    """Transformer encoder for aggregating a target node and its neighbors.

    The model takes a sequence-like tensor containing the representation of a
    target node followed by its multi-hop or sampled neighbor representations.
    Stacked encoder layers refine the sequence, then an attention layer learns
    how much each neighbor contributes to the target representation. The final
    output is the target embedding combined with the attention-weighted neighbor
    embedding.

    Parameters
    ----------
    hops : int
        Number of neighbor hops represented in the input sequence. The expected
        sequence length is ``hops + 1``: one target node plus its neighbors.
    input_dim : int
        Input feature dimension kept for compatibility with the training
        pipeline. The current implementation expects the input tensor features
        to already match ``hidden_dim``.
    n_layers : int, default=6
        Number of transformer encoder layers.
    num_heads : int, default=8
        Number of attention heads in each encoder layer.
    hidden_dim : int, default=64
        Hidden feature dimension used by the transformer layers.
    ffn_dim : int, default=64
        Feed-forward dimension argument kept for API compatibility. Internally
        the model uses ``2 * hidden_dim``.
    dropout_rate : float, default=0.0
        Dropout probability applied after self-attention and feed-forward
        blocks.
    attention_dropout_rate : float, default=0.1
        Dropout probability applied to attention weights.

    Notes
    -----
    ``forward(batched_data)`` expects a tensor with shape
    ``(batch_size, hops + 1, hidden_dim)`` and returns a tensor with shape
    ``(batch_size, hidden_dim)`` for batched inputs.
    """

    def __init__(
            self,
            hops,
            input_dim,
            n_layers=6,
            num_heads=8,
            hidden_dim=64,
            ffn_dim=64,
            dropout_rate=0.0,
            attention_dropout_rate=0.1
    ):
        super().__init__()

        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        tensor = batched_data

        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        output = self.final_ln(tensor)

        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)
        node_tensor, neighbor_tensor = split_tensor[0], split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
