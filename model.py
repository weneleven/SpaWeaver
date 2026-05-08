import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF(nn.Module):
    """Compute a multi-scale radial basis function kernel matrix.

    ``RBF`` is a ``torch.nn.Module`` used to convert a feature matrix into a
    sample-by-sample Gaussian kernel similarity matrix. It computes pairwise
    squared Euclidean distances with ``torch.cdist`` and evaluates several
    bandwidth scales. The kernel matrices from all scales are summed.

    Parameters
    ----------
    n_kernels : int, default=5
        Number of Gaussian kernels with different bandwidth scales.
    mul_factor : float, default=2.0
        Multiplicative factor used to space adjacent bandwidth scales.
    bandwidth : float or None, default=None
        Fixed base bandwidth. If ``None``, the base bandwidth is estimated from
        the pairwise squared distance matrix during ``forward``.

    Attributes
    ----------
    bandwidth_multipliers : torch.Tensor
        Tensor of shape ``(n_kernels,)`` containing the bandwidth scale
        multipliers.
    bandwidth : float or None
        Fixed base bandwidth, or ``None`` when batch-wise bandwidth estimation
        is used.

    Notes
    -----
    ``forward`` expects ``X`` to be a ``torch.Tensor`` with shape
    ``(n_samples, n_features)``. ``bandwidth_multipliers`` is moved to
    ``X.device`` inside ``forward`` before kernel values are computed.

    Examples
    --------
    >>> import torch
    >>> from SpaWeaver.model import RBF
    >>> kernel = RBF(n_kernels=3)
    >>> X = torch.randn(4, 8)
    >>> K = kernel(X)
    >>> K.shape
    torch.Size([4, 4])
    """

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        """Initialize the multi-scale RBF kernel.

        Parameters
        ----------
        n_kernels : int, default=5
            Number of Gaussian kernels with different bandwidth scales.
        mul_factor : float, default=2.0
            Multiplicative factor used to construct
            ``bandwidth_multipliers`` as
            ``mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)``.
        bandwidth : float or None, default=None
            Fixed base bandwidth. If ``None``, ``get_bandwidth`` estimates the
            bandwidth from the pairwise squared distance matrix passed during
            ``forward``.

        """
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """Return the bandwidth used by the RBF kernel.

        Parameters
        ----------
        L2_distances : torch.Tensor
            Pairwise squared Euclidean distance matrix with shape
            ``(n_samples, n_samples)``.

        Returns
        -------
        float or torch.Tensor
            If ``self.bandwidth`` is not ``None``, returns that fixed value.
            Otherwise returns ``L2_distances.data.sum() / (n_samples ** 2 -
            n_samples)``.

        Notes
        -----
        This method assumes ``L2_distances`` is already on the device used by
        the current ``forward`` call. The diagonal entries produced by
        ``torch.cdist(X, X) ** 2`` are zero.
        """
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        """Compute the summed multi-scale RBF kernel matrix.

        Parameters
        ----------
        X : torch.Tensor
            Input feature matrix with shape ``(n_samples, n_features)``. The
            tensor should contain numeric features and must be a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Kernel matrix with shape ``(n_samples, n_samples)``. Entry
            ``(i, j)`` is the summed multi-scale RBF similarity between samples
            ``X[i]`` and ``X[j]``.

        Notes
        -----
        ``bandwidth_multipliers`` is moved to ``X.device`` inside this method.
        The pairwise distance matrix, estimated bandwidth, and output kernel are
        computed on the same device as ``X``.
        """
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
    """Aggregate a target representation with neighboring representations.

    ``transformerModel`` is a ``torch.nn.Module`` that applies stacked
    transformer encoder layers to a fixed-length sequence. The first sequence
    position is treated as the target node and the remaining positions are
    treated as neighbors. After the encoder stack, the model computes
    target-conditioned attention weights over the neighbors and adds the
    weighted neighbor summary to the encoded target representation.

    Parameters
    ----------
    hops : int
        Number of neighbor positions represented in the input sequence. The
        model sets ``seq_len = hops + 1``.
    input_dim : int
        Input feature dimension stored as an attribute. The current
        implementation does not project from ``input_dim`` and expects the last
        dimension of ``batched_data`` to match ``hidden_dim``.
    n_layers : int, default=6
        Number of encoder layers.
    num_heads : int, default=8
        Number of attention heads in each encoder layer.
    hidden_dim : int, default=64
        Hidden feature dimension used by attention, layer normalization,
        feed-forward layers, and target-neighbor aggregation.
    ffn_dim : int, default=64
        Constructor argument kept for API compatibility. The implementation
        sets ``self.ffn_dim = 2 * hidden_dim`` regardless of this value.
    dropout_rate : float, default=0.0
        Dropout probability applied after self-attention and feed-forward
        blocks.
    attention_dropout_rate : float, default=0.1
        Dropout probability applied inside multi-head attention.

    Attributes
    ----------
    seq_len : int
        Expected sequence length, equal to ``hops + 1``.
    input_dim : int
        Stored input feature dimension.
    hidden_dim : int
        Hidden feature dimension.
    ffn_dim : int
        Feed-forward dimension used by encoder layers, equal to
        ``2 * hidden_dim`` in the current implementation.
    num_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    layers : torch.nn.ModuleList
        List of ``EncoderLayer`` modules.
    final_ln : torch.nn.LayerNorm
        Final layer normalization applied to the encoded sequence.
    attn_layer : torch.nn.Linear
        Linear layer mapping concatenated target-neighbor features of size
        ``2 * hidden_dim`` to one attention score.
    scaling : torch.nn.Parameter
        Learnable scalar initialized to ``0.5``. It is defined by the module but
        is not used in ``forward``.

    Notes
    -----
    ``forward`` expects ``batched_data`` to be a ``torch.Tensor`` whose last
    dimension is ``hidden_dim`` and whose sequence length is ``hops + 1``. All
    model parameters and input tensors must be on compatible devices, following
    standard PyTorch module behavior.

    Examples
    --------
    >>> import torch
    >>> from SpaWeaver.model import transformerModel
    >>> model = transformerModel(hops=3, input_dim=64, hidden_dim=64)
    >>> x = torch.randn(2, 4, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 64])
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
        """Initialize the transformer aggregation model.

        Parameters
        ----------
        hops : int
            Number of neighbor positions represented in the input sequence.
            The model expects sequence length ``hops + 1`` in ``forward``.
        input_dim : int
            Input feature dimension stored as ``self.input_dim``. The current
            code does not use this value to build an input projection.
        n_layers : int, default=6
            Number of ``EncoderLayer`` blocks.
        num_heads : int, default=8
            Number of attention heads in each ``EncoderLayer``.
        hidden_dim : int, default=64
            Hidden feature dimension. ``batched_data`` passed to ``forward`` is
            expected to have this size in its last dimension.
        ffn_dim : int, default=64
            API-compatible argument. The implementation sets ``self.ffn_dim`` to
            ``2 * hidden_dim`` and does not otherwise use this argument.
        dropout_rate : float, default=0.0
            Dropout probability passed to each ``EncoderLayer`` for residual
            dropout after attention and feed-forward blocks.
        attention_dropout_rate : float, default=0.1
            Dropout probability passed to multi-head attention.

        """
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
        """Encode a target-neighbor sequence.

        Parameters
        ----------
        batched_data : torch.Tensor
            Input tensor with expected shape ``(batch_size, hops + 1,
            hidden_dim)``. The first sequence position is interpreted as the
            target representation. The remaining ``hops`` positions are
            interpreted as neighboring representations. The tensor should be on
            the same device as the model parameters.

        Returns
        -------
        torch.Tensor
            Aggregated target representation. For input with ``batch_size > 1``
            and no other singleton dimensions, the expected shape is
            ``(batch_size, hidden_dim)``. The implementation calls ``squeeze()``
            before returning, so singleton dimensions, including a batch
            dimension of size ``1``, may be removed.

        Notes
        -----
        The method first applies all encoder layers and ``final_ln``. It then
        computes neighbor attention by concatenating the encoded target token
        with each encoded neighbor token, applying ``attn_layer``, and
        normalizing over the neighbor dimension with ``softmax``.
        """
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
